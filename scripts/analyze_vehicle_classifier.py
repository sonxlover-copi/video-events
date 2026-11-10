"""Разобрать ошибки классификатора по классам, размерам объектов и трекам."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from train_vehicle_classifier import CLASSES, _build_model


def _scores(confusion: np.ndarray) -> dict[str, Any]:
    """Посчитать точность и F1 по матрице ошибок в фиксированном порядке классов."""
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
                "support": expected,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
    total = int(confusion.sum())
    return {
        "samples": total,
        "accuracy": round(float(np.trace(confusion) / total), 4) if total else None,
        "macro_f1": round(float(np.mean([item["f1"] for item in per_class])), 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }


def analyze(
    dataset_path: Path, checkpoint_path: Path, output_dir: Path, device_name: str
) -> dict[str, Any]:
    """Сопоставить CNN с простым baseline и собрать примеры для ручной проверки."""
    import torch

    dataset = np.load(dataset_path)
    required = {"validation_images", "validation_labels", "validation_metadata"}
    if not required.issubset(dataset.files):
        raise ValueError(
            "В датасете нет provenance-полей; повторно выполни команду prepare"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("classes", ())) != CLASSES:
        raise ValueError("Список классов checkpoint не совпадает с анализатором")

    device = torch.device(device_name)
    model = _build_model(torch)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    images = torch.from_numpy(
        dataset["validation_images"].astype(np.float32) / 255.0
    ).permute(0, 3, 1, 2)
    labels = dataset["validation_labels"].astype(np.int64)
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(images), 256):
            batch = images[start : start + 256].to(device)
            probabilities.append(model(batch).softmax(1).cpu().numpy())
    predictions = np.concatenate(probabilities, axis=0)

    metadata = [json.loads(value) for value in dataset["validation_metadata"]]
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    size_groups: dict[str, list[int]] = defaultdict(list)
    for index, (label, scores, item) in enumerate(
        zip(labels, predictions, metadata, strict=True)
    ):
        predicted = int(scores.argmax())
        confusion[int(label), predicted] += 1
        groups[(item["sequence"], str(item["target_id"]))].append(index)
        width, height = item["bbox"][2:]
        short_side = min(width, height)
        size_name = (
            "20-31 px"
            if short_side < 32
            else "32-63 px"
            if short_side < 64
            else "64+ px"
        )
        size_groups[size_name].append(index)

    # Один объект получает один голос: усредняем вероятности его повторных кадров.
    track_confusion = np.zeros_like(confusion)
    track_predictions: list[tuple[str, str, int, int, float]] = []
    inconsistent_tracks = 0
    for (sequence, target_id), indexes in groups.items():
        track_labels = {int(labels[index]) for index in indexes}
        if len(track_labels) != 1:
            inconsistent_tracks += 1
            continue
        expected = track_labels.pop()
        mean_scores = predictions[indexes].mean(axis=0)
        predicted = int(mean_scores.argmax())
        track_confusion[expected, predicted] += 1
        track_predictions.append(
            (sequence, target_id, expected, predicted, float(mean_scores[predicted]))
        )

    training_counts = np.bincount(
        dataset["train_labels"].astype(np.int64), minlength=len(CLASSES)
    )
    training_tracks: dict[tuple[str, str], int] = {}
    for index, value in enumerate(dataset["train_metadata"]):
        item = json.loads(value)
        training_tracks[(item["sequence"], str(item["target_id"]))] = int(
            dataset["train_labels"][index]
        )
    training_track_counts = np.bincount(
        list(training_tracks.values()), minlength=len(CLASSES)
    )
    baseline_class = int(training_counts.argmax())
    baseline_track_class = int(training_track_counts.argmax())
    baseline_confusion = np.zeros_like(confusion)
    for label in labels:
        baseline_confusion[int(label), baseline_class] += 1
    baseline_track_confusion = np.zeros_like(track_confusion)
    for _, _, expected, _, _ in track_predictions:
        baseline_track_confusion[expected, baseline_track_class] += 1

    size_summary = {}
    for size_name, indexes in size_groups.items():
        group_confusion = np.zeros_like(confusion)
        for index in indexes:
            group_confusion[int(labels[index]), int(predictions[index].argmax())] += 1
        size_summary[size_name] = _scores(group_confusion)

    output_dir.mkdir(parents=True, exist_ok=True)
    wrong_indexes = [
        index
        for index, label in enumerate(labels)
        if predictions[index].argmax() != label
    ]
    wrong_indexes.sort(key=lambda index: float(predictions[index].max()), reverse=True)
    wrong_indexes = wrong_indexes[:30]
    _save_review_grid(
        dataset["validation_images"],
        labels,
        predictions,
        metadata,
        wrong_indexes,
        output_dir / "misclassified_grid.png",
    )

    repeated_tracks = sum(len(indexes) > 1 for indexes in groups.values())
    report = {
        "classes": list(CLASSES),
        "validation_split": "UA-DETRAC Train, 12 held-out sequences",
        "validation_by_crop": _scores(confusion),
        "validation_by_track": _scores(track_confusion),
        "track_sampling": {
            "unique_tracks": len(groups),
            "tracks_with_multiple_crops": repeated_tracks,
            "crops_per_track_mean": round(len(labels) / len(groups), 3),
            "tracks_with_inconsistent_labels": inconsistent_tracks,
            "train_tracks_by_class": {
                class_name: int(training_track_counts[index])
                for index, class_name in enumerate(CLASSES)
            },
        },
        "majority_class_baseline_by_crop": {
            "predicted_class": CLASSES[baseline_class],
            **_scores(baseline_confusion),
            "note": "Always predicts the most common class in the balanced training sample.",
        },
        "majority_class_baseline_by_track": {
            "predicted_class": CLASSES[baseline_track_class],
            **_scores(baseline_track_confusion),
            "note": "Always predicts the most common vehicle class among unique training tracks.",
        },
        "crop_short_side_by_size": size_summary,
        "largest_confusions": {
            f"{CLASSES[row]}_as_{CLASSES[column]}": int(confusion[row, column])
            for row in range(len(CLASSES))
            for column in range(len(CLASSES))
            if row != column
        },
        "review_grid": str(output_dir / "misclassified_grid.png"),
        "limitations": [
            "Track-level probabilities average repeated sampled frames from one annotated trajectory.",
            "The crops come from ground-truth boxes; detector misses and box errors are excluded.",
            "The majority baseline is a deliberately simple reference, not a competing trained model.",
        ],
    }
    (output_dir / "error_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _save_review_grid(
    images: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: list[dict[str, Any]],
    indexes: list[int],
    output_path: Path,
) -> None:
    """Сохранить локальную сетку ошибок с исходным видео и ID траектории."""
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = (
        ImageFont.truetype(str(font_path), 12)
        if font_path.is_file()
        else ImageFont.load_default()
    )
    cell_width, cell_height = 300, 116
    columns = 4
    rows = max(1, (len(indexes) + columns - 1) // columns)
    canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    draw = ImageDraw.Draw(canvas)
    for position, index in enumerate(indexes):
        x = (position % columns) * cell_width
        y = (position // columns) * cell_height
        crop = Image.fromarray(images[index]).resize((96, 96), Image.Resampling.NEAREST)
        canvas.paste(crop, (x, y))
        predicted = int(probabilities[index].argmax())
        item = metadata[index]
        lines = [
            f"true {CLASSES[int(labels[index])]} -> {CLASSES[predicted]}",
            f"confidence {probabilities[index][predicted]:.2f}",
            f"{item['sequence']} / track {item['target_id']}",
            f"frame {item['frame']} / box {item['bbox'][2]:.0f}x{item['bbox'][3]:.0f}",
        ]
        for line_number, line in enumerate(lines):
            draw.text(
                (x + 102, y + 4 + line_number * 20), line, font=font, fill="black"
            )
    canvas.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/processed/vehicle-classifier/dataset.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "data/processed/vehicle-classifier/vehicle_type_classifier.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "runs/vehicle-classifier/analysis"
    )
    parser.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    args = parser.parse_args()
    try:
        report = analyze(args.dataset, args.checkpoint, args.output_dir, args.device)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
