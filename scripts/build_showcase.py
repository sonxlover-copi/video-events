"""Оформить результат запуска для портфолио, сохранив время появления событий."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def build_showcase(run_dir: Path, output_path: Path, duration_s: float) -> None:
    """Добавить панель событий, которые уже подтверждены к текущему кадру."""
    if output_path.exists():
        raise ValueError(f"Output already exists: {output_path}")
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with (run_dir / "events.csv").open(encoding="utf-8", newline="") as stream:
        events = sorted(
            csv.DictReader(stream), key=lambda row: int(row["confirmation_frame_index"])
        )
    capture = cv2.VideoCapture(str(run_dir / summary["annotated_video"]))
    fps = summary["video"]["output_fps"]
    if not capture.isOpened():
        raise ValueError("Could not open annotated video")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = output_path.with_name(f".{output_path.stem}.tmp.mp4")
    writer = cv2.VideoWriter(
        str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1600, 900)
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"Could not create video: {intermediate}")
    event_index = 0
    recent = []
    try:
        for frame_index in range(round(duration_s * fps)):
            ok, frame = capture.read()
            if not ok:
                break
            while (
                event_index < len(events)
                and int(events[event_index]["confirmation_frame_index"]) <= frame_index
            ):
                event = events[event_index]
                if event.get("plate_text"):
                    recent.append(event)
                event_index += 1
            canvas = np.full((900, 1600, 3), (24, 19, 14), dtype=np.uint8)
            cv2.putText(
                canvas,
                "VIDEO EVENTS",
                (32, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.35,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Vehicle tracking / line crossings / local plate OCR",
                (34, 106),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.74,
                (185, 175, 160),
                1,
                cv2.LINE_AA,
            )
            canvas[150:798, 32:1184] = cv2.resize(
                frame, (1152, 648), interpolation=cv2.INTER_AREA
            )
            cv2.rectangle(canvas, (1208, 150), (1568, 798), (39, 31, 23), -1)
            cv2.putText(
                canvas,
                "CROSSINGS",
                (1230, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (185, 175, 160),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                str(event_index),
                (1230, 252),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.65,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "PLATES AT CROSSING",
                (1230, 297),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (185, 175, 160),
                1,
                cv2.LINE_AA,
            )
            # Используем снимок события, а не итоговый каталог: номер, прочитанный
            # в конце клипа, не должен появиться в начале демонстрации.
            for index, event in enumerate(reversed(recent[-4:])):
                y = 345 + index * 103
                cv2.putText(
                    canvas,
                    event["plate_text"],
                    (1230, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (145, 225, 95),
                    2,
                    cv2.LINE_AA,
                )
                detail = f"ID {event['track_id']} / {event['class_name']} / {event['timestamp_s']}s"
                cv2.putText(
                    canvas,
                    detail,
                    (1230, y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (210, 205, 195),
                    1,
                    cv2.LINE_AA,
                )
            if not recent:
                cv2.putText(
                    canvas,
                    "Waiting for a plate event",
                    (1230, 350),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.47,
                    (185, 175, 160),
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                canvas,
                "YOLO11 + ByteTrack + ONNX OCR | Actual processing output",
                (32, 841),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (220, 215, 205),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Footage: Mike Bird / Pexels 2103099 | Source video plays at its original speed",
                (32, 875),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (160, 155, 145),
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()
        capture.release()
    # H.264 и faststart позволяют смотреть клип в браузере без полной загрузки.
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(intermediate),
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        check=True,
    )
    intermediate.unlink()


def main() -> None:
    """Позволить собрать отдельный фрагмент из уже готовых видео и CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=12)
    args = parser.parse_args()
    if not 0 < args.duration_s < float("inf"):
        parser.error("--duration-s must be a positive finite number")
    build_showcase(args.run_dir, args.output, args.duration_s)


if __name__ == "__main__":
    main()
