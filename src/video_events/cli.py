"""CLI для проверки видео и подсчёта транспорта на одной камере."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from video_events.attributes import (
    ATTRIBUTE_FIELDS,
    AttributeOptions,
    VehicleAttributes,
)
from video_events.config import CounterConfig, load_config
from video_events.crossing import LineCrossingCounter
from video_events.plates import PlateReader, prepare_plate_models

# Путь по умолчанию привязан к репозиторию, а не к текущей папке терминала.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "traffic.yaml"
EVENT_FIELDS = (
    "event_id",
    "confirmation_frame_index",
    "crossing_frame_estimate",
    "timestamp_s",
    "track_id",
    "class_name",
    "confidence",
    "direction",
    "line_intersection_x_px",
    "line_intersection_y_px",
)
DIRECTIONS = ("negative_to_positive", "positive_to_negative")
FALLBACK_FPS = 25.0


def _video_capture(path: Path) -> cv2.VideoCapture:
    """Открыть файл и сообщить понятную ошибку до начала обработки кадров."""
    if not path.is_file():
        raise ValueError(f"Input video does not exist or is not a file: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open the video: {path}")
    return capture


def inspect_video(path: Path) -> dict[str, Any]:
    """Вернуть свойства видео и убедиться, что первый кадр декодируется."""
    capture = _video_capture(path)
    try:
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Video opened but no frame could be decoded: {path}")
        height, width = frame.shape[:2]
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_number = int(capture.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(
            chr((fourcc_number >> (8 * index)) & 0xFF) for index in range(4)
        )
        # Некоторые контейнеры не сообщают FPS; duration в таком случае неизвестна.
        fps_valid = math.isfinite(fps) and fps > 0
        duration = reported_frames / fps if fps_valid and reported_frames > 0 else None
        return {
            "file": path.name,
            "width": width,
            "height": height,
            "fps": round(fps, 3) if math.isfinite(fps) and fps > 0 else None,
            "reported_frame_count": reported_frames if reported_frames > 0 else None,
            "duration_s": round(duration, 3) if duration is not None else None,
            "fourcc": fourcc.strip("\x00 "),
            "first_frame_decoded": True,
        }
    finally:
        capture.release()


def _resolve_device(device: str) -> str:
    """Преобразовать `auto` в CUDA-устройство или CPU для Ultralytics."""
    if device.casefold() != "auto":
        return device
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def _class_maps(
    model: YOLO, requested: tuple[str, ...]
) -> tuple[list[int], dict[int, str]]:
    """Сопоставить имена из YAML с числовыми ID классов конкретной модели."""
    names = model.names
    if isinstance(names, list):
        name_by_id = {index: str(name) for index, name in enumerate(names)}
    else:
        name_by_id = {int(index): str(name) for index, name in names.items()}
    id_by_name = {name.casefold(): index for index, name in name_by_id.items()}

    unknown = [name for name in requested if name.casefold() not in id_by_name]
    if unknown:
        available = ", ".join(name_by_id[index] for index in sorted(name_by_id))
        raise ValueError(
            f"Model does not contain requested classes {unknown}. Available classes: {available}"
        )

    class_ids = [id_by_name[name.casefold()] for name in requested]
    selected_names = {class_id: name_by_id[class_id] for class_id in class_ids}
    return class_ids, selected_names


def _line_pixels(
    config: CounterConfig,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Перевести независимые от разрешения координаты YAML в пиксели кадра."""
    start = (
        round(config.line_start[0] * (width - 1)),
        round(config.line_start[1] * (height - 1)),
    )
    end = (
        round(config.line_end[0] * (width - 1)),
        round(config.line_end[1] * (height - 1)),
    )
    return start, end


def _draw_overlay(
    frame: Any,
    line: tuple[tuple[int, int], tuple[int, int]],
    counts: dict[tuple[str, str], int],
    event: dict[str, int | float | str] | None,
) -> None:
    """Нанести на кадр линию, текущий счёт и маркер нового события."""
    start, end = line
    cv2.line(frame, start, end, (0, 255, 255), 2, cv2.LINE_AA)
    height, width = frame.shape[:2]
    rows = [
        f"{class_name}: {sum(counts[(class_name, direction)] for direction in DIRECTIONS)}"
        for class_name in sorted({key[0] for key in counts})
    ]
    panel_height = max(36, 28 + 24 * len(rows))
    cv2.rectangle(
        frame,
        (8, 8),
        (min(width - 8, 310), min(height - 8, panel_height)),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        "Line crossings",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for index, row in enumerate(rows):
        cv2.putText(
            frame,
            row,
            (18, 56 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (80, 230, 80),
            1,
            cv2.LINE_AA,
        )
    if event is not None:
        center = (
            int(float(event["line_intersection_x_px"])),
            int(float(event["line_intersection_y_px"])),
        )
        label = f"{event['class_name']} {event['direction']}"
        cv2.circle(frame, center, 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            label,
            (max(4, center[0] + 8), max(20, center[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )


def _encode_h264(video_path: Path) -> str:
    """Перекодировать промежуточный MP4 в H.264 для лучшей совместимости плееров."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("FFmpeg is required to create a broadly compatible H.264 MP4")

    temporary_path = video_path.with_name(f".{video_path.stem}.h264.tmp.mp4")
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        temporary_path.unlink(missing_ok=True)
        details = (
            (exc.stderr or "").strip()
            if isinstance(exc, subprocess.CalledProcessError)
            else str(exc)
        )
        raise ValueError(
            f"FFmpeg could not encode '{video_path}' as H.264: {details}"
        ) from exc

    # Сначала готовим соседний файл, затем атомарно подменяем промежуточный ролик.
    os.replace(temporary_path, video_path)
    return "h264"


def count_video(
    input_path: Path,
    config: CounterConfig,
    output_dir: Path,
    attribute_options: AttributeOptions | None = None,
) -> dict[str, Any]:
    """Обработать видео целиком и сохранить видео с разметкой и отчёты."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    options = attribute_options or AttributeOptions()
    if shutil.which("ffmpeg") is None:
        raise ValueError("FFmpeg is required. Install it before processing the video.")

    annotated_path = output_dir / "annotated.mp4"
    events_path = output_dir / "events.csv"
    summary_path = output_dir / "summary.json"
    vehicles_path = output_dir / "vehicles.csv"
    # Не перезаписываем предыдущий запуск: проще выбрать новую пустую папку.
    existing_outputs = [
        path.name
        for path in (annotated_path, events_path, summary_path, vehicles_path)
        if path.exists()
    ]
    if existing_outputs:
        raise ValueError(
            f"Output files already exist in '{output_dir}': {', '.join(existing_outputs)}. "
            "Choose an empty output directory to preserve earlier results."
        )
    if input_path in {
        annotated_path.resolve(),
        events_path.resolve(),
        summary_path.resolve(),
        vehicles_path.resolve(),
    }:
        raise ValueError("Input video must not be one of the output files")

    # Проверяем готовность OCR до создания выходных файлов. Во время обработки
    # модели не скачиваются: их заранее подготавливает команда prepare-plates.
    attribute_started = time.perf_counter()
    attributes = VehicleAttributes(options) if options.enabled else None
    attribute_init_s = time.perf_counter() - attribute_started
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = _video_capture(input_path)
    writer: cv2.VideoWriter | None = None
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise ValueError(
                f"Video opened but no frame could be decoded: {input_path}"
            )
        height, width = first_frame.shape[:2]
        if width < 32 or height < 32:
            raise ValueError(
                f"Video frame is too small for annotation: {width}x{height}"
            )
        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps_valid = math.isfinite(source_fps) and source_fps > 0
        # Для редких файлов с неизвестным FPS выбираем явный fallback и записываем это в отчёт.
        output_fps = source_fps if fps_valid else FALLBACK_FPS
        reported_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Веса загружаются один раз; затем tracker сохраняет состояние между кадрами.
        device = _resolve_device(config.device)
        load_started = time.perf_counter()
        model = YOLO(str(config.model))
        model_load_s = time.perf_counter() - load_started
        class_ids, class_names = _class_maps(model, config.classes)

        line = _line_pixels(config, width, height)
        counter = LineCrossingCounter(
            start=(float(line[0][0]), float(line[0][1])),
            end=(float(line[1][0]), float(line[1][1])),
            deadband_px=config.line_deadband_px,
        )

        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError(
                f"Could not create '{annotated_path}'. Check that this OpenCV build can encode MP4."
            )

        # Нулевые значения нужны, чтобы сводка показывала и классы без событий.
        selected_class_names = [class_names[class_id] for class_id in class_ids]
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for class_name in selected_class_names:
            for direction in DIRECTIONS:
                counts[(class_name, direction)] = 0
        events_written = 0
        frames_processed = 0
        process_started = time.perf_counter()
        with events_path.open("w", newline="", encoding="utf-8") as events_file:
            event_fields = (
                EVENT_FIELDS + ATTRIBUTE_FIELDS if attributes else EVENT_FIELDS
            )
            event_writer = csv.DictWriter(events_file, fieldnames=event_fields)
            event_writer.writeheader()

            # Первый кадр уже прочитан выше; дальше читаем следующий только после
            # завершения детекции, подсчёта и записи текущего.
            frame = first_frame
            while frame is not None:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=config.tracker,
                    conf=config.confidence_threshold,
                    imgsz=config.image_size,
                    classes=class_ids,
                    device=device,
                    verbose=False,
                )
                result = results[0]
                annotated = result.plot(labels=attributes is None)
                frame_event = None
                attribute_labels = []
                boxes = result.boxes
                # Без track ID нельзя отличить новое событие от повторной детекции.
                if boxes is not None and boxes.id is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.detach().cpu().tolist()
                    track_ids = boxes.id.detach().cpu().tolist()
                    class_indices = boxes.cls.detach().cpu().tolist()
                    confidences = boxes.conf.detach().cpu().tolist()
                    for box, track_id, class_index, confidence in zip(
                        xyxy, track_ids, class_indices, confidences, strict=True
                    ):
                        class_id = int(class_index)
                        class_name = class_names.get(class_id)
                        if class_name is None:
                            continue
                        center = (
                            (float(box[0]) + float(box[2])) / 2,
                            (float(box[1]) + float(box[3])) / 2,
                        )
                        features = None
                        if attributes is not None:
                            features = attributes.update(
                                frame, box, int(track_id), class_name, frames_processed
                            )
                            attribute_labels.append(
                                (box, int(track_id), class_name, features)
                            )
                        event = counter.update(
                            track_id=int(track_id),
                            class_name=class_name,
                            confidence=float(confidence),
                            center=center,
                            frame_index=frames_processed,
                            timestamp_s=frames_processed / output_fps,
                        )
                        if event is None:
                            continue
                        events_written += 1
                        event_row = {"event_id": events_written, **event}
                        if features is not None:
                            # Снимок относится к моменту события. Более поздние
                            # наблюдения доступны отдельно в vehicles.csv.
                            event_row.update(features)
                        event_writer.writerow(event_row)
                        counts[(class_name, str(event["direction"]))] += 1
                        frame_event = event

                # Подтверждённые номера рисуем последними: подписи соседних
                # рамок не должны закрывать основной результат распознавания.
                for box, track_id, class_name, features in sorted(
                    attribute_labels, key=lambda item: bool(item[3]["plate_text"])
                ):
                    _draw_vehicle_attributes(
                        annotated, box, track_id, class_name, features
                    )
                _draw_overlay(annotated, line, counts, frame_event)
                writer.write(annotated)
                frames_processed += 1
                ok, frame = capture.read()
                if not ok:
                    frame = None

        # Закрываем кодировщик до замера времени, чтобы включить финальную запись MP4.
        writer.release()
        writer = None
        video_codec = _encode_h264(annotated_path)
        processing_s = time.perf_counter() - process_started
        if frames_processed == 0:
            raise ValueError(f"No frames were processed from: {input_path}")

        # Сводка хранит оба направления отдельно и общий итог по классу.
        totals = {
            class_name: {
                direction: counts[(class_name, direction)] for direction in DIRECTIONS
            }
            for class_name in selected_class_names
        }
        for class_counts in totals.values():
            class_counts["total"] = sum(
                class_counts[direction] for direction in DIRECTIONS
            )

        vehicle_rows = attributes.vehicle_rows(output_fps) if attributes else []
        if attributes is not None:
            vehicle_fields = (
                "track_id",
                "class_name",
                "first_seen_s",
                "last_seen_s",
                "plate_attempts",
                *ATTRIBUTE_FIELDS,
            )
            with vehicles_path.open("w", newline="", encoding="utf-8") as stream:
                vehicle_writer = csv.DictWriter(stream, fieldnames=vehicle_fields)
                vehicle_writer.writeheader()
                vehicle_writer.writerows(vehicle_rows)

        summary = {
            "schema_version": 2 if attributes else 1,
            "input_file": input_path.name,
            "video": {
                "width": width,
                "height": height,
                "source_fps": round(source_fps, 3) if fps_valid else None,
                "output_fps": round(output_fps, 3),
                "fps_fallback_used": not fps_valid,
                "reported_frame_count": reported_frame_count
                if reported_frame_count > 0
                else None,
                "processed_frame_count": frames_processed,
                "processed_duration_s": round(frames_processed / output_fps, 3),
                "codec": video_codec,
            },
            "detector": {
                "model": config.model.name,
                "tracker": config.tracker,
                "device": device,
                "classes": selected_class_names,
                "confidence_threshold": config.confidence_threshold,
                "image_size": config.image_size,
            },
            "line": {
                "start_normalized": list(config.line_start),
                "end_normalized": list(config.line_end),
                "deadband_px": config.line_deadband_px,
                "direction_definition": (
                    "negative_to_positive is the change from negative to positive signed distance "
                    "relative to the oriented line start-to-end"
                ),
            },
            "events": {
                "total": events_written,
                "by_class": totals,
                "csv": events_path.name,
            },
            "performance": {
                "model_load_s": round(model_load_s, 3),
                "processing_s": round(processing_s, 3),
                "end_to_end_fps_excluding_model_load": (
                    round(frames_processed / processing_s, 3)
                    if processing_s > 0
                    else None
                ),
                "note": "Processing time includes video decode, inference, overlay rendering, and encoding.",
            },
            "annotated_video": annotated_path.name,
        }
        if attributes is not None:
            summary["attributes"] = {
                **asdict(options),
                "initialization_s": round(attribute_init_s, 3),
                "vehicles_csv": vehicles_path.name,
                "tracked_ids": len(vehicle_rows),
                "by_color": dict(Counter(row["vehicle_color"] for row in vehicle_rows)),
                "confirmed_plates": sum(
                    row["plate_status"] == "confirmed" for row in vehicle_rows
                ),
                "plate_models": attributes.reader.model_info()
                if attributes.reader
                else None,
                "event_values": "observations available at confirmation_frame_index",
                "vehicle_values": "final observations for each tracker ID",
            }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def _draw_vehicle_attributes(
    frame: Any,
    box: list[float],
    track_id: int,
    class_name: str,
    features: dict[str, str | int | float],
) -> None:
    """Нарисовать компактную подпись, не меняя исходный кадр для распознавания."""
    label = f"ID {track_id} {class_name} / {features['vehicle_color']}"
    if features["plate_text"]:
        label += f" / {features['plate_text']}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
    )
    height, width = frame.shape[:2]
    x = max(0, min(round(box[0]), width - text_width - 8))
    y = max(text_height + 6, min(round(box[1]) - 5, height - baseline - 3))
    cv2.rectangle(
        frame,
        (x, y - text_height - 5),
        (min(width - 1, x + text_width + 6), y + baseline + 2),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def read_plate_image(
    input_path: Path, output_dir: Path, plate_format: str, min_confidence: float
) -> dict[str, Any]:
    """Прочитать номера на фотографии без трекинга и контрольной линии."""
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    image_path, report_path = output_dir / "plates.jpg", output_dir / "plates.json"
    if image_path.exists() or report_path.exists():
        raise ValueError(
            "Plate output files already exist. Choose a new output directory."
        )
    if not input_path.is_file():
        raise ValueError(f"Input image does not exist: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not decode image: {input_path}")
    reader = PlateReader(plate_format, min_confidence)
    readings = reader.read(image)
    for reading in readings:
        x1, y1, x2, y2 = reading.box
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 230, 80), 2)
        cv2.putText(
            image,
            reading.text,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 230, 80),
            2,
            cv2.LINE_AA,
        )
    report = {
        "input_file": input_path.name,
        "plates": [asdict(reading) for reading in readings],
        "status": "single_image_candidates" if readings else "not_read",
        "models": reader.model_info(),
        "annotated_image": image_path.name,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), image):
        raise ValueError(f"Could not save annotated image: {image_path}")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    """Описать команды CLI и их обязательные аргументы."""
    parser = argparse.ArgumentParser(
        prog="video-events",
        description="Inspect a video or count vehicle crossings of one virtual line.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect a video file and decode its first frame"
    )
    inspect_parser.add_argument(
        "--input", type=Path, required=True, help="input video file"
    )

    count_parser = subparsers.add_parser(
        "count", help="count vehicle line crossings in an offline video"
    )
    count_parser.add_argument(
        "--input", type=Path, required=True, help="input video file"
    )
    count_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"YAML configuration (default: {DEFAULT_CONFIG})",
    )
    count_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for annotated.mp4, events.csv, and summary.json",
    )
    count_parser.add_argument(
        "--colors", action="store_true", help="estimate vehicle body colors"
    )
    count_parser.add_argument(
        "--plates", action="store_true", help="read and confirm license plates"
    )
    count_parser.add_argument(
        "--attribute-interval",
        type=int,
        default=10,
        help="sample each track every N frames (default: 10)",
    )
    count_parser.add_argument(
        "--plate-min-observations",
        type=int,
        default=2,
        help="matching readings required to confirm a plate (at least 2)",
    )
    count_parser.add_argument(
        "--plate-min-confidence",
        type=float,
        default=0.85,
        help="minimum mean OCR confidence (default: 0.85)",
    )
    count_parser.add_argument(
        "--plate-format",
        choices=("generic", "ru"),
        default="generic",
        help="generic Latin plates or standard Russian car plates",
    )
    prepare_parser = subparsers.add_parser(
        "prepare-plates",
        help="download and validate local plate detection and OCR models",
    )
    prepare_parser.add_argument(
        "--force", action="store_true", help="download the plate models again"
    )
    image_parser = subparsers.add_parser(
        "read-plates", help="read plates from a single image"
    )
    image_parser.add_argument("--input", type=Path, required=True)
    image_parser.add_argument("--output-dir", type=Path, required=True)
    image_parser.add_argument(
        "--plate-format", choices=("generic", "ru"), default="generic"
    )
    image_parser.add_argument("--plate-min-confidence", type=float, default=0.85)
    return parser


def main() -> int:
    """Разобрать аргументы, выполнить команду и вернуть код завершения."""
    args = _parser().parse_args()
    try:
        if args.command == "inspect":
            result = inspect_video(args.input.expanduser().resolve())
        elif args.command == "prepare-plates":
            result = prepare_plate_models(force=args.force)
        elif args.command == "read-plates":
            result = read_plate_image(
                args.input,
                args.output_dir,
                args.plate_format,
                args.plate_min_confidence,
            )
        else:
            options = AttributeOptions(
                colors=args.colors,
                plates=args.plates,
                interval=args.attribute_interval,
                plate_min_confidence=args.plate_min_confidence,
                plate_min_observations=args.plate_min_observations,
                plate_format=args.plate_format,
            )
            config = load_config(args.config)
            result = count_video(args.input, config, args.output_dir, options)
    except (TypeError, ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
