"""Подготовить и обучить классификатор типов транспорта на UA-DETRAC.

Выборка делится по целым видеопоследовательностям: соседние кадры одной камеры
не попадают одновременно в обучение и проверку. Исходные архивы остаются нетронутыми.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DEFAULT_ANNOTATIONS = ROOT / "data/raw/ua_detrac/DETRAC-Train-Annotations-XML.zip"
DEFAULT_IMAGES = ROOT / "data/raw/ua_detrac/DETRAC-Images.zip"
DEFAULT_OUTPUT = ROOT / "data/processed/vehicle-classifier"
CLASSES = ("car", "bus", "van")
IMAGE_SIZE = 64


def _read_sequences(archive: zipfile.ZipFile) -> dict[str, bytes]:
    """Загрузить XML из архива и проверить, что в нём есть последовательности."""
    annotations = {
        Path(name).stem: archive.read(name)
        for name in archive.namelist()
        if name.lower().endswith(".xml")
    }
    if not annotations:
        raise ValueError("В архиве разметки не найдены XML-файлы")
    return annotations


def _sequence_counts(annotations: dict[str, bytes]) -> dict[str, Counter[str]]:
    """Посчитать размеченные объекты каждого класса в каждой сцене."""
    counts: dict[str, Counter[str]] = {}
    for sequence, xml_bytes in annotations.items():
        sequence_counts: Counter[str] = Counter()
        root = ElementTree.fromstring(xml_bytes)
        for target in root.iter("target"):
            attributes = target.find("attribute")
            if attributes is not None:
                vehicle_type = attributes.attrib.get("vehicle_type", "").casefold()
                if vehicle_type in CLASSES:
                    sequence_counts[vehicle_type] += 1
        counts[sequence] = sequence_counts
    return counts


def _split_sequences(
    counts: dict[str, Counter[str]], validation_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    """Подобрать фиксированный holdout сцен с близким распределением классов."""
    if len(counts) < 5:
        raise ValueError("Для разбиения по сценам нужно не менее пяти XML-файлов")
    total = Counter()
    for sequence_counts in counts.values():
        total.update(sequence_counts)
    if any(total[name] == 0 for name in CLASSES):
        raise ValueError("В разметке отсутствует один или несколько выбранных классов")

    sequence_names = sorted(counts)
    validation_size = max(1, round(len(sequence_names) * validation_fraction))
    rng = random.Random(seed)
    best_score: float | None = None
    best_validation: set[str] | None = None

    # Несколько случайных кандидатов позволяют учитывать редкие классы,
    # не превращая выбор сцен в ручной список, подогнанный под метрики.
    for _ in range(4000):
        candidate = set(rng.sample(sequence_names, validation_size))
        observed = Counter()
        for sequence in candidate:
            observed.update(counts[sequence])
        score = sum(
            ((observed[name] / total[name]) - validation_fraction) ** 2
            for name in CLASSES
        )
        if all(observed[name] > 0 for name in CLASSES) and (
            best_score is None or score < best_score
        ):
            best_score = score
            best_validation = candidate

    if best_validation is None:
        raise ValueError("Не удалось собрать holdout со всеми тремя классами")
    training = set(sequence_names) - best_validation
    return training, best_validation


def _audit_annotations(
    annotations: dict[str, bytes], image_width: int, image_height: int
) -> dict[str, Any]:
    """Проверить рамки на некорректные координаты и записать объём проблем."""
    totals: Counter[str] = Counter()
    class_totals: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for sequence, xml_bytes in annotations.items():
        root = ElementTree.fromstring(xml_bytes)
        for frame in root.findall("frame"):
            frame_number = int(frame.attrib.get("num", "0"))
            target_list = frame.find("target_list")
            if target_list is None:
                continue
            for target in target_list.findall("target"):
                totals["targets"] += 1
                attrs = target.find("attribute")
                box = target.find("box")
                name = (
                    attrs.attrib.get("vehicle_type", "unknown").casefold()
                    if attrs is not None
                    else "unknown"
                )
                class_totals[name] += 1
                if box is None:
                    issue = "missing_box"
                else:
                    try:
                        left = float(box.attrib["left"])
                        top = float(box.attrib["top"])
                        width = float(box.attrib["width"])
                        height = float(box.attrib["height"])
                        values = (left, top, width, height)
                        if not np.isfinite(values).all():
                            issue = "non_finite"
                        elif width <= 0 or height <= 0:
                            issue = "non_positive_size"
                        elif (
                            left < 0
                            or top < 0
                            or left + width > image_width
                            or top + height > image_height
                        ):
                            issue = "partially_outside_frame"
                        else:
                            issue = ""
                    except (KeyError, ValueError):
                        issue = "invalid_coordinates"
                if issue:
                    totals[issue] += 1
                    if len(examples) < 30:
                        examples.append(
                            {
                                "sequence": sequence,
                                "frame": frame_number,
                                "target_id": target.attrib.get("id"),
                                "class": name,
                                "issue": issue,
                            }
                        )
    return {
        "frame_size": [image_width, image_height],
        "target_count": totals["targets"],
        "class_counts": dict(sorted(class_totals.items())),
        "issue_counts": {
            key: value for key, value in sorted(totals.items()) if key != "targets"
        },
        "issue_examples": examples,
    }


def prepare_dataset(
    annotation_archive: Path,
    image_archive: Path,
    output_dir: Path,
    *,
    frame_stride: int,
    validation_fraction: float,
    seed: int,
    max_per_class: int,
) -> dict[str, Any]:
    """Собрать сбалансированные RGB-кропы и отчёт об аудите разметки."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(annotation_archive) as archive:
        annotations = _read_sequences(archive)
    sequence_counts = _sequence_counts(annotations)
    training_sequences, validation_sequences = _split_sequences(
        sequence_counts, validation_fraction, seed
    )

    # UA-DETRAC хранит кадры в 960x540; размер проверяется на первом кадре.
    with zipfile.ZipFile(image_archive) as images:
        first_frame = next(
            (
                name
                for name in images.namelist()
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            ),
            None,
        )
        if first_frame is None:
            raise ValueError("В архиве изображений не найдены кадры")
        with images.open(first_frame) as stream:
            image_width, image_height = Image.open(stream).size

        audit = _audit_annotations(annotations, image_width, image_height)
        selected_sequences = training_sequences | validation_sequences
        selected_frames: dict[
            str,
            dict[
                int,
                list[
                    tuple[
                        str,
                        str,
                        tuple[int, int, int, int],
                        tuple[float, float, float, float],
                    ]
                ],
            ],
        ] = defaultdict(dict)
        for sequence in sorted(selected_sequences):
            root = ElementTree.fromstring(annotations[sequence])
            for frame in root.findall("frame"):
                frame_number = int(frame.attrib.get("num", "0"))
                if frame_number < 1 or (frame_number - 1) % frame_stride:
                    continue
                target_list = frame.find("target_list")
                if target_list is None:
                    continue
                boxes = []
                for target in target_list.findall("target"):
                    attributes = target.find("attribute")
                    box = target.find("box")
                    if attributes is None or box is None:
                        continue
                    class_name = attributes.attrib.get("vehicle_type", "").casefold()
                    if class_name not in CLASSES:
                        continue
                    try:
                        left = float(box.attrib["left"])
                        top = float(box.attrib["top"])
                        width = float(box.attrib["width"])
                        height = float(box.attrib["height"])
                    except (KeyError, ValueError):
                        continue
                    if (
                        not np.isfinite((left, top, width, height)).all()
                        or width < 20
                        or height < 20
                    ):
                        continue
                    x1, y1 = max(0, round(left)), max(0, round(top))
                    x2 = min(image_width, round(left + width))
                    y2 = min(image_height, round(top + height))
                    if x2 - x1 < 16 or y2 - y1 < 16:
                        continue
                    boxes.append(
                        (
                            class_name,
                            target.attrib.get("id", ""),
                            (x1, y1, x2, y2),
                            (left, top, width, height),
                        )
                    )
                selected_frames[sequence][frame_number] = boxes

        # Reservoir sampling ограничивает размер датасета и не отдаёт
        # преимущество ранним сценам внутри архива.
        rng = random.Random(seed)
        samples: dict[str, dict[str, list[np.ndarray]]] = {
            "train": {name: [] for name in CLASSES},
            "validation": {name: [] for name in CLASSES},
        }
        sample_metadata: dict[str, dict[str, list[dict[str, Any]]]] = {
            "train": {name: [] for name in CLASSES},
            "validation": {name: [] for name in CLASSES},
        }
        seen: Counter[tuple[str, str]] = Counter()
        image_names = set(images.namelist())
        frame_items = sorted(
            (sequence, frame_number)
            for sequence, frames in selected_frames.items()
            for frame_number in frames
        )
        needed_names = {
            f"DETRAC-Images/{sequence}/img{frame_number:05d}.jpg"
            for sequence, frame_number in frame_items
        }
        missing_frames = needed_names - image_names
        if missing_frames:
            raise ValueError(
                f"В архиве отсутствуют выбранные кадры, например: {min(missing_frames)}"
            )

        for sequence, frame_number in frame_items:
            member = f"DETRAC-Images/{sequence}/img{frame_number:05d}.jpg"
            with images.open(member) as stream:
                frame_image = Image.open(stream).convert("RGB")
                if frame_image.size != (image_width, image_height):
                    raise ValueError(f"Непостоянный размер кадра: {member}")
                for (
                    class_name,
                    target_id,
                    (x1, y1, x2, y2),
                    original_box,
                ) in selected_frames[sequence][frame_number]:
                    split_name = (
                        "train" if sequence in training_sequences else "validation"
                    )
                    bucket = samples[split_name][class_name]
                    key = (split_name, class_name)
                    seen[key] += 1
                    crop = np.asarray(
                        frame_image.crop((x1, y1, x2, y2)).resize(
                            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
                        ),
                        dtype=np.uint8,
                    )
                    metadata = {
                        "sequence": sequence,
                        "frame": frame_number,
                        "target_id": target_id,
                        "bbox": [round(value, 2) for value in original_box],
                    }
                    if len(bucket) < max_per_class:
                        bucket.append(crop)
                        sample_metadata[split_name][class_name].append(metadata)
                    else:
                        replacement = rng.randrange(seen[key])
                        if replacement < max_per_class:
                            bucket[replacement] = crop
                            sample_metadata[split_name][class_name][replacement] = (
                                metadata
                            )

    arrays: dict[str, np.ndarray] = {}
    split_summary: dict[str, dict[str, int]] = {}
    for split_name, class_samples in samples.items():
        images_list: list[np.ndarray] = []
        labels: list[int] = []
        metadata_list: list[dict[str, Any]] = []
        for label, class_name in enumerate(CLASSES):
            images_list.extend(class_samples[class_name])
            labels.extend([label] * len(class_samples[class_name]))
            metadata_list.extend(sample_metadata[split_name][class_name])
        if not images_list or any(not class_samples[name] for name in CLASSES):
            raise ValueError(
                f"После фильтрации в {split_name} не хватает примеров одного из классов"
            )
        order = list(range(len(labels)))
        random.Random(seed + (0 if split_name == "train" else 1)).shuffle(order)
        arrays[f"{split_name}_images"] = np.stack(
            [images_list[index] for index in order]
        )
        arrays[f"{split_name}_labels"] = np.asarray(
            [labels[index] for index in order], dtype=np.int64
        )
        arrays[f"{split_name}_metadata"] = np.asarray(
            [json.dumps(metadata_list[index], ensure_ascii=False) for index in order]
        )
        split_summary[split_name] = {
            class_name: len(class_samples[class_name]) for class_name in CLASSES
        }

    dataset_path = output_dir / "dataset.npz"
    np.savez_compressed(dataset_path, **arrays)
    report = {
        "dataset": "UA-DETRAC Train",
        "annotation_archive": str(annotation_archive),
        "image_archive": str(image_archive),
        "seed": seed,
        "frame_stride": frame_stride,
        "validation_fraction": validation_fraction,
        "image_size": IMAGE_SIZE,
        "classes": list(CLASSES),
        "excluded_class": {
            "others": "Редкая категория; исключена из этого эксперимента."
        },
        "split_sequences": {
            "train": sorted(training_sequences),
            "validation": sorted(validation_sequences),
        },
        "sample_counts": split_summary,
        "annotation_audit": audit,
        "dataset_file": str(dataset_path),
    }
    (output_dir / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _build_model(torch: Any) -> Any:
    """Создать компактную CNN без внешних весов и скрытой загрузки файлов."""
    from torch import nn

    class ConvBlock(nn.Module):
        def __init__(
            self, in_channels: int, out_channels: int, stride: int = 1
        ) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 3, stride=stride, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    groups=out_channels,
                    bias=False,
                ),
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, images: Any) -> Any:
            return self.layers(images)

    class VehicleTypeClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                ConvBlock(3, 24, stride=2),
                ConvBlock(24, 40, stride=2),
                ConvBlock(40, 72, stride=2),
                ConvBlock(72, 96, stride=2),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(96, len(CLASSES)),
            )

        def forward(self, images: Any) -> Any:
            return self.features(images)

    return VehicleTypeClassifier()


def _group_tracks(metadata_values: np.ndarray) -> dict[tuple[str, str], list[int]]:
    """Сгруппировать повторные кадры одной размеченной траектории."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, value in enumerate(metadata_values):
        item = json.loads(value)
        groups[(item["sequence"], str(item["target_id"]))].append(index)
    return groups


def _confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Собрать матрицу ошибок в порядке классов, заданном в начале файла."""
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return confusion


def _track_confusion_matrix(
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: dict[tuple[str, str], list[int]],
) -> np.ndarray:
    """Дать каждой траектории один голос после усреднения её кадров."""
    track_labels = []
    track_predictions = []
    for indexes in groups.values():
        expected_classes = set(labels[indexes].tolist())
        if len(expected_classes) != 1:
            raise ValueError("Один track_id содержит разные XML-классы")
        track_labels.append(expected_classes.pop())
        track_predictions.append(int(probabilities[indexes].mean(axis=0).argmax()))
    return _confusion_matrix(
        np.asarray(track_labels, dtype=np.int64),
        np.asarray(track_predictions, dtype=np.int64),
    )


def _metric_summary(confusion: np.ndarray) -> dict[str, Any]:
    """Посчитать accuracy и macro F1 для выбора эпохи и финального отчёта."""
    per_class = []
    for index, class_name in enumerate(CLASSES):
        true_positive = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        expected = int(confusion[index, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / expected if expected else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    total = int(confusion.sum())
    return {
        "samples": total,
        "accuracy": float(np.trace(confusion) / total) if total else None,
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def train_model(
    dataset_path: Path,
    output_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Обучить собственную CNN и сохранить метрики на holdout-сценах."""
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from torchvision import transforms

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(device_name)
    data = np.load(dataset_path)

    def make_tensors(split: str) -> tuple[Any, Any]:
        images = torch.from_numpy(data[f"{split}_images"].astype(np.float32) / 255.0)
        images = images.permute(0, 3, 1, 2).contiguous()
        labels = torch.from_numpy(data[f"{split}_labels"].astype(np.int64))
        return images, labels

    train_images, train_labels = make_tensors("train")
    validation_images, validation_labels = make_tensors("validation")
    required_metadata = {"train_metadata", "validation_metadata"}
    if not required_metadata.issubset(data.files):
        raise ValueError("Датасет не содержит track metadata; повтори prepare")
    train_groups = _group_tracks(data["train_metadata"])
    validation_groups = _group_tracks(data["validation_metadata"])
    class_counts = torch.bincount(train_labels, minlength=len(CLASSES)).float()
    train_labels_numpy = train_labels.numpy()
    tracks_by_class = np.zeros(len(CLASSES), dtype=np.int64)
    for indexes in train_groups.values():
        track_class = int(train_labels_numpy[indexes[0]])
        if any(train_labels_numpy[index] != track_class for index in indexes):
            raise ValueError("Один train track содержит разные XML-классы")
        tracks_by_class[track_class] += 1
    sample_weights = np.zeros(len(train_labels_numpy), dtype=np.float64)
    for indexes in train_groups.values():
        track_class = int(train_labels_numpy[indexes[0]])
        per_sample_weight = 1.0 / (tracks_by_class[track_class] * len(indexes))
        sample_weights[indexes] = per_sample_weight
    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels),
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        TensorDataset(validation_images, validation_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = _build_model(torch).to(device)
    augment = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.12, hue=0.02
            ),
        ]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    history: list[dict[str, float | int]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Аугментации меняют освещение и ракурс, но сохраняют класс объекта.
            images = augment(images).clamp_(0.0, 1.0)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.03)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)

        model.eval()
        validation_loss = 0.0
        probability_batches = []
        with torch.inference_mode():
            for images, labels in validation_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                validation_loss += F.cross_entropy(logits, labels).item() * len(labels)
                probability_batches.append(logits.softmax(dim=1).cpu().numpy())
        validation_probabilities = np.concatenate(probability_batches)
        validation_label_values = validation_labels.numpy()
        crop_confusion = _confusion_matrix(
            validation_label_values, validation_probabilities.argmax(axis=1)
        )
        track_confusion = _track_confusion_matrix(
            validation_label_values, validation_probabilities, validation_groups
        )
        crop_metrics = _metric_summary(crop_confusion)
        track_metrics = _metric_summary(track_confusion)
        # При подборе эпохи каждая машина имеет один голос вне зависимости от
        # числа кадров, на которых она присутствует в разметке.
        macro_f1 = track_metrics["macro_f1"]
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_images),
            "validation_loss": validation_loss / len(validation_images),
            "crop_validation_accuracy": crop_metrics["accuracy"],
            "crop_validation_macro_f1": crop_metrics["macro_f1"],
            "track_validation_accuracy": track_metrics["accuracy"],
            "track_validation_macro_f1": macro_f1,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Обучение не сохранило ни одной модели")
    model.load_state_dict(best_state)
    model.eval()
    probability_batches = []
    with torch.inference_mode():
        for images, labels in validation_loader:
            logits = model(images.to(device))
            probability_batches.append(logits.softmax(dim=1).cpu().numpy())
    validation_probabilities = np.concatenate(probability_batches)
    validation_label_values = validation_labels.numpy()
    crop_confusion = _confusion_matrix(
        validation_label_values, validation_probabilities.argmax(axis=1)
    )
    track_confusion = _track_confusion_matrix(
        validation_label_values, validation_probabilities, validation_groups
    )
    crop_metrics = _metric_summary(crop_confusion)
    track_metrics = _metric_summary(track_confusion)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "vehicle_type_classifier.pt"
    torch.save(
        {
            "state_dict": best_state,
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "architecture": "depthwise-separable CNN, 24/40/72/96 channels",
            "seed": seed,
            "best_track_validation_macro_f1": best_macro_f1,
        },
        checkpoint_path,
    )
    report = {
        "task": "vehicle crop classification",
        "dataset": "UA-DETRAC Train",
        "split_unit": "video sequence",
        "classes": list(CLASSES),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "architecture": "depthwise-separable CNN, 24/40/72/96 channels",
        "input_size": [IMAGE_SIZE, IMAGE_SIZE],
        "training_examples": len(train_images),
        "validation_examples": len(validation_images),
        "training_tracks": len(train_groups),
        "validation_tracks": len(validation_groups),
        "training_tracks_by_class": dict(
            zip(CLASSES, tracks_by_class.tolist(), strict=True)
        ),
        "class_counts_train": dict(
            zip(CLASSES, class_counts.int().tolist(), strict=True)
        ),
        "class_counts_validation": dict(
            zip(
                CLASSES,
                torch.bincount(validation_labels, minlength=len(CLASSES)).tolist(),
                strict=True,
            )
        ),
        "seed": seed,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "selection_metric": "track-level validation macro F1 (mean probabilities per trajectory)",
        "best_validation_macro_f1": best_macro_f1,
        "validation_by_crop": crop_metrics,
        "validation_by_track": track_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "limitations": [
            "Это классификация кропов по готовым XML-боксам, а не обучение детектора.",
            "Все сцены относятся к UA-DETRAC Train; перенос на другие камеры не проверен.",
            "Разметка содержит шум и частичные перекрытия объектов.",
            "Результаты не сравнивались с коммерческим сценарием или edge-платой.",
        ],
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare", help="Проверить разметку и подготовить кропы"
    )
    prepare.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    prepare.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    prepare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    prepare.add_argument("--frame-stride", type=int, default=10)
    prepare.add_argument("--validation-fraction", type=float, default=0.2)
    prepare.add_argument("--max-per-class", type=int, default=2500)
    prepare.add_argument("--seed", type=int, default=42)

    train = subparsers.add_parser(
        "train", help="Обучить и оценить собственный классификатор"
    )
    train.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT / "dataset.npz")
    train.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            if args.frame_stride < 1 or args.max_per_class < 1:
                raise ValueError(
                    "frame-stride и max-per-class должны быть положительными"
                )
            if not 0.05 <= args.validation_fraction <= 0.5:
                raise ValueError("validation-fraction должна быть от 0.05 до 0.5")
            report = prepare_dataset(
                args.annotations,
                args.images,
                args.output_dir,
                frame_stride=args.frame_stride,
                validation_fraction=args.validation_fraction,
                seed=args.seed,
                max_per_class=args.max_per_class,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
                raise ValueError(
                    "epochs, batch-size и learning-rate должны быть положительными"
                )
            report = train_model(
                args.dataset,
                args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device_name=args.device,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        parser.error(str(exc))
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
