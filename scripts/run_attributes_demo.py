"""Проверить цвет, OCR и события на панорамировании контрольной фотографии."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from video_events.attributes import AttributeOptions
from video_events.cli import count_video
from video_events.config import CounterConfig

EXAMPLE_URL = (
    "https://raw.githubusercontent.com/ankandrew/fast-alpr/master/assets/test_image.png"
)


def main() -> None:
    """Создать короткий входной ролик и запустить настоящий пайплайн анализа."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=Path("data/samples/fast-alpr-test.png")
    )
    parser.add_argument(
        "--download-example",
        action="store_true",
        help="download the upstream FastALPR example if missing",
    )
    parser.add_argument("--model", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/attributes/photo-pan")
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError("Demo directory already exists. Choose a new --output-dir.")
    if not args.model.is_file():
        raise ValueError(f"Vehicle detector weights not found: {args.model}")
    if not args.image.is_file() and args.download_example:
        args.image.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(EXAMPLE_URL, args.image)
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(
            "Provide --image or pass --download-example for the upstream sample."
        )
    # Это намеренно контролируемое движение фотографии, а не новый датасет.
    # Размер номера сохраняется: мы проверяем связку модулей без дорисовки букв.
    image = cv2.resize(image, (518, 331))
    args.output_dir.mkdir(parents=True)
    video_path = args.output_dir / "photo-pan.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (800, 450)
    )
    if not writer.isOpened():
        raise ValueError(f"Could not create demo video: {video_path}")
    try:
        for index in range(96):
            frame = np.full((450, 800, 3), (25, 30, 35), dtype=np.uint8)
            x = round(30 + 220 * index / 95)
            frame[65:396, x : x + 518] = image
            cv2.putText(
                frame,
                "PHOTO PAN / integration check",
                (20, 433),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()
    config = CounterConfig(
        model=args.model.resolve(),
        device="auto",
        classes=("car",),
        confidence_threshold=0.25,
        image_size=640,
        tracker="bytetrack.yaml",
        line_deadband_px=8,
        line_start=(0.5, 0.12),
        line_end=(0.5, 0.94),
    )
    summary = count_video(
        video_path,
        config,
        args.output_dir / "result",
        AttributeOptions(colors=True, plates=True, interval=6),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
