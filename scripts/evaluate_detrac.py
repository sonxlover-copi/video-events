"""Сравнить события счётчика с траекториями из XML-аннотаций UA-DETRAC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree

from video_events.crossing import LineCrossingCounter

if TYPE_CHECKING:
    from video_events.config import CounterConfig

DIRECTIONS = ("negative_to_positive", "positive_to_negative")


def _line_pixels(
    config: CounterConfig,
    width: int,
    height: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Повторить округление координат из CLI, чтобы проверять ту же линию."""
    start = (
        round(config.line_start[0] * (width - 1)),
        round(config.line_start[1] * (height - 1)),
    )
    end = (
        round(config.line_end[0] * (width - 1)),
        round(config.line_end[1] * (height - 1)),
    )
    return start, end


def _ground_truth_events(
    annotation_path: Path,
    config: CounterConfig,
    width: int,
    height: int,
    fps: float,
    class_aliases: dict[str, str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Преобразовать XML-треки в события пересечения той же линии."""
    try:
        root = ElementTree.parse(annotation_path).getroot()
    except ElementTree.ParseError as exc:
        raise ValueError(f"Invalid XML annotation '{annotation_path}': {exc}") from exc

    line_start, line_end = _line_pixels(config, width, height)
    counter = LineCrossingCounter(
        start=(float(line_start[0]), float(line_start[1])),
        end=(float(line_end[0]), float(line_end[1])),
        deadband_px=config.line_deadband_px,
    )
    aliases = {
        source.casefold(): target.casefold()
        for source, target in (class_aliases or {}).items()
    }
    selected_classes = {name.casefold() for name in config.classes}
    source_classes = selected_classes | {
        source for source, target in aliases.items() if target in selected_classes
    }
    events: list[dict[str, Any]] = []

    # XML-категории и ID относятся к ручной разметке, а не к предсказаниям YOLO.
    for frame in root.findall("frame"):
        frame_index = int(frame.attrib["num"]) - 1
        target_list = frame.find("target_list")
        if target_list is None:
            continue

        for target in target_list.findall("target"):
            attributes = target.find("attribute")
            box = target.find("box")
            if attributes is None or box is None:
                continue

            source_class = attributes.attrib.get("vehicle_type", "").casefold()
            if source_class not in source_classes:
                continue
            class_name = aliases.get(source_class, source_class)

            left = float(box.attrib["left"])
            top = float(box.attrib["top"])
            box_width = float(box.attrib["width"])
            box_height = float(box.attrib["height"])
            center = (left + box_width / 2, top + box_height / 2)
            event = counter.update(
                track_id=int(target.attrib["id"]),
                class_name=class_name,
                confidence=1.0,
                center=center,
                frame_index=frame_index,
                timestamp_s=frame_index / fps,
            )
            if event is not None:
                events.append(event)

    return root.attrib.get("name", annotation_path.stem), events


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    """Прочитать и привести CSV-события к типам, удобным для сравнения."""
    required_fields = {
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
    }
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing_fields = required_fields - set(reader.fieldnames or ())
        if missing_fields:
            raise ValueError(
                f"Events CSV is missing columns: {', '.join(sorted(missing_fields))}"
            )
        predictions = []
        for row in reader:
            predictions.append(
                {
                    "event_id": int(row["event_id"]),
                    "confirmation_frame_index": int(row["confirmation_frame_index"]),
                    "crossing_frame_estimate": float(row["crossing_frame_estimate"]),
                    "timestamp_s": float(row["timestamp_s"]),
                    "track_id": int(row["track_id"]),
                    "class_name": row["class_name"].casefold(),
                    "confidence": float(row["confidence"]),
                    "direction": row["direction"],
                    "line_intersection_x_px": float(row["line_intersection_x_px"]),
                    "line_intersection_y_px": float(row["line_intersection_y_px"]),
                }
            )
    return predictions


def _match_events(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    tolerance_s: float,
    spatial_tolerance_px: float | None = None,
) -> list[tuple[int, int]]:
    """Сопоставить события по классу, направлению, времени и близости на линии."""
    candidates: dict[int, list[int]] = {}
    for expected_index, event in enumerate(expected):
        candidates[expected_index] = sorted(
            (
                predicted_index
                for predicted_index, prediction in enumerate(predicted)
                if prediction["class_name"] == event["class_name"]
                and prediction["direction"] == event["direction"]
                and abs(prediction["timestamp_s"] - event["timestamp_s"]) <= tolerance_s
                and (
                    spatial_tolerance_px is None
                    or math.hypot(
                        prediction["line_intersection_x_px"]
                        - event["line_intersection_x_px"],
                        prediction["line_intersection_y_px"]
                        - event["line_intersection_y_px"],
                    )
                    <= spatial_tolerance_px
                )
            ),
            key=lambda index: abs(
                predicted[index]["timestamp_s"] - event["timestamp_s"]
            ),
        )

    # Переназначение пары нужно, когда несколько соседних событий попадают
    # в одно временное окно; так одно предсказание не засчитывается дважды.
    prediction_to_expected: dict[int, int] = {}

    def assign(expected_index: int, visited_predictions: set[int]) -> bool:
        for predicted_index in candidates[expected_index]:
            if predicted_index in visited_predictions:
                continue
            visited_predictions.add(predicted_index)
            previous_expected = prediction_to_expected.get(predicted_index)
            if previous_expected is None or assign(
                previous_expected, visited_predictions
            ):
                prediction_to_expected[predicted_index] = expected_index
                return True
        return False

    # Сначала сопоставляем события с наименьшим числом возможных пар.
    for expected_index in sorted(
        candidates,
        key=lambda index: (len(candidates[index]), expected[index]["timestamp_s"]),
    ):
        assign(expected_index, set())

    return sorted(
        (expected_index, predicted_index)
        for predicted_index, expected_index in prediction_to_expected.items()
    )


def _counts(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    matches: list[tuple[int, int]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, int | float | None]:
    """Посчитать TP/FP/FN и метрики для выбранной группы событий."""
    expected_indexes = {
        index for index, event in enumerate(expected) if predicate(event)
    }
    predicted_indexes = {
        index for index, event in enumerate(predicted) if predicate(event)
    }
    true_positives = sum(
        1
        for expected_index, predicted_index in matches
        if expected_index in expected_indexes and predicted_index in predicted_indexes
    )
    false_positives = len(predicted_indexes) - true_positives
    false_negatives = len(expected_indexes) - true_positives
    precision = true_positives / len(predicted_indexes) if predicted_indexes else None
    recall = true_positives / len(expected_indexes) if expected_indexes else None
    # F1 считаем из целых TP/FP/FN, чтобы округление precision/recall не влияло
    # на результат. Для полностью пустой группы метрика не определена.
    f1_denominator = 2 * true_positives + false_positives + false_negatives
    f1 = 2 * true_positives / f1_denominator if f1_denominator else None
    count_error = len(predicted_indexes) - len(expected_indexes)
    return {
        "ground_truth": len(expected_indexes),
        "predicted": len(predicted_indexes),
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "count_error": count_error,
        "absolute_count_error": abs(count_error),
        "relative_count_error": (
            round(count_error / len(expected_indexes), 4) if expected_indexes else None
        ),
    }


def evaluate(
    *,
    video_path: Path,
    config_path: Path,
    annotation_path: Path,
    events_path: Path,
    tolerance_s: float,
    spatial_tolerance_px: float | None = None,
    class_aliases: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Собрать воспроизводимый отчёт по одному видео и его XML-аннотации."""
    # Геометрию и сопоставление можно проверять без OpenCV, весов и PyTorch.
    # Тяжёлые зависимости нужны только при оценке настоящего видео.
    from video_events.cli import inspect_video
    from video_events.config import load_config

    config = load_config(config_path)
    selected_classes = {name.casefold() for name in config.classes}
    invalid_alias_targets = {
        target.casefold()
        for target in (class_aliases or {}).values()
        if target.casefold() not in selected_classes
    }
    if invalid_alias_targets:
        raise ValueError(
            "Class alias targets must be configured detector classes: "
            + ", ".join(sorted(invalid_alias_targets))
        )
    video_info = inspect_video(video_path.expanduser().resolve())
    fps = video_info["fps"] or 25.0
    width, height = video_info["width"], video_info["height"]
    sequence_name, expected = _ground_truth_events(
        annotation_path, config, width, height, fps, class_aliases
    )
    predicted = _read_predictions(events_path)
    matches = _match_events(expected, predicted, tolerance_s, spatial_tolerance_px)
    line_start, line_end = _line_pixels(config, width, height)

    directions = {
        direction: _counts(
            expected,
            predicted,
            matches,
            lambda event, selected=direction: event["direction"] == selected,
        )
        for direction in DIRECTIONS
    }
    class_directions = {
        f"{class_name}/{direction}": _counts(
            expected,
            predicted,
            matches,
            lambda event, selected_class=class_name, selected_direction=direction: (
                event["class_name"] == selected_class
                and event["direction"] == selected_direction
            ),
        )
        for class_name in (name.casefold() for name in config.classes)
        for direction in DIRECTIONS
    }

    matched_expected = {expected_index for expected_index, _ in matches}
    matched_predictions = {predicted_index for _, predicted_index in matches}
    return {
        "sequence": sequence_name,
        "data_role": "development clip; use a separate sequence for final evaluation",
        "video": {
            "file": video_path.name,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": video_info["reported_frame_count"],
        },
        "evaluation": {
            "classes": list(config.classes),
            "tolerance_s": tolerance_s,
            "spatial_tolerance_px": spatial_tolerance_px,
            "class_aliases": class_aliases or {},
            "line_start_px": list(line_start),
            "line_end_px": list(line_end),
            "deadband_px": config.line_deadband_px,
            "matching": (
                "one-to-one by class and direction within time and spatial tolerances"
            ),
            "ground_truth": "derived from manually annotated XML tracks",
        },
        "overall": _counts(expected, predicted, matches, lambda event: bool(event)),
        "by_direction": directions,
        "by_class_and_direction": class_directions,
        "missed_events": [
            event
            for index, event in enumerate(expected)
            if index not in matched_expected
        ],
        "extra_events": [
            event
            for index, event in enumerate(predicted)
            if index not in matched_predictions
        ],
    }


def _parser() -> argparse.ArgumentParser:
    """Описать входы отчёта, чтобы его можно было повторить на другом видео."""
    parser = argparse.ArgumentParser(
        description="Compare predicted line-crossing events with UA-DETRAC XML tracks."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--tolerance-s", type=float, default=1.0)
    parser.add_argument(
        "--spatial-tolerance-px",
        type=float,
        default=40.0,
        help="maximum distance between predicted and annotated line intersections",
    )
    parser.add_argument(
        "--class-alias",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="map an annotation class to a detector class, for example van=car",
    )
    parser.add_argument("--report", type=Path)
    return parser


def main() -> int:
    """Запустить оценку и вывести JSON в терминал или файл."""
    args = _parser().parse_args()
    if args.tolerance_s < 0:
        print("error: --tolerance-s must be non-negative", file=sys.stderr)
        return 2
    if args.spatial_tolerance_px < 0:
        print("error: --spatial-tolerance-px must be non-negative", file=sys.stderr)
        return 2

    class_aliases = {}
    for item in args.class_alias:
        source, separator, target = item.partition("=")
        if not separator or not source.strip() or not target.strip():
            print(
                f"error: invalid --class-alias '{item}'; expected SOURCE=TARGET",
                file=sys.stderr,
            )
            return 2
        class_aliases[source.strip()] = target.strip()

    try:
        report = evaluate(
            video_path=args.video,
            config_path=args.config,
            annotation_path=args.annotations,
            events_path=args.events,
            tolerance_s=args.tolerance_s,
            spatial_tolerance_px=args.spatial_tolerance_px,
            class_aliases=class_aliases,
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
