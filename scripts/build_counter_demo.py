"""Собрать иллюстрацию работы счётчика без видео, весов и детектора."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from video_events.crossing import LineCrossingCounter


def build_demo(output_dir: Path) -> None:
    """Нарисовать заданные траектории и сохранить реальные события счётчика."""
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = 800, 450
    fps, frame_count = 12, 96
    counter = LineCrossingCounter((400, 185), (400, 390), deadband_px=8)
    font = ImageFont.load_default(size=17)
    title_font = ImageFont.load_default(size=25)
    frames = []
    events = []
    totals = {"negative_to_positive": 0, "positive_to_negative": 0}
    # Верхний трек пересекает продолжение прямой, но не сам отрезок.
    # Два других проходят через линию навстречу друг другу и дают два события.
    trajectories = (
        (1, 145, (118, 162, 187), False, 0.0),
        (2, 245, (84, 211, 170), False, 0.08),
        (3, 345, (244, 184, 96), True, -0.08),
    )
    for frame_index in range(frame_count):
        image = Image.new("RGB", (width, height), (15, 23, 36))
        draw = ImageDraw.Draw(image)
        draw.text(
            (26, 20), "VIDEO EVENTS / LINE COUNTER", font=title_font, fill="white"
        )
        draw.text(
            (26, 58),
            "Synthetic tracks - no detector or tracking benchmark",
            font=font,
            fill=(166, 181, 202),
        )
        draw.rounded_rectangle((24, 104, 776, 404), radius=14, fill=(30, 43, 59))
        for y in (195, 295):
            for x in range(38, 760, 52):
                draw.line((x, y, x + 26, y), fill=(76, 91, 110), width=2)
        draw.line((400, 110, 400, 184), fill=(76, 91, 110), width=2)
        draw.line((400, 185, 400, 390), fill=(250, 215, 100), width=3)
        draw.ellipse((395, 180, 405, 190), fill=(250, 215, 100))
        draw.ellipse((395, 385, 405, 395), fill=(250, 215, 100))
        for track_id, y, color, reverse, offset in trajectories:
            progress = min(1.0, max(0.0, frame_index / (frame_count - 1) + offset))
            x = 95 + 610 * (1 - progress if reverse else progress)
            event = counter.update(
                track_id=track_id,
                class_name="car",
                confidence=1.0,
                center=(x, y),
                frame_index=frame_index,
                timestamp_s=frame_index / fps,
            )
            if event is not None:
                events.append({"event_id": len(events) + 1, **event})
                totals[str(event["direction"])] += 1
            draw.rounded_rectangle(
                (x - 29, y - 15, x + 29, y + 15), radius=7, fill=color
            )
            draw.rectangle((x - 34, y - 20, x + 34, y + 20), outline=color, width=2)
            label = (
                f"car / ID {track_id}" if track_id != 1 else "ID 1 / outside segment"
            )
            draw.text((max(35, x - 34), y - 43), label, font=font, fill=color)
        # Направления подписаны в терминах движения на этой иллюстрации.
        draw.text(
            (26, 419),
            f"Left to right: {totals['positive_to_negative']}",
            font=font,
            fill="white",
        )
        draw.text(
            (280, 419),
            f"Right to left: {totals['negative_to_positive']}",
            font=font,
            fill="white",
        )
        draw.text((600, 419), f"Events: {len(events)}", font=font, fill=(250, 215, 100))
        frames.append(image)

    frames[72].save(output_dir / "counter-demo.png")
    frames[0].save(
        output_dir / "counter-demo.gif",
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
    )
    with (output_dir / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(events[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(events)


def main() -> None:
    """Позволить пересобрать демонстрацию одной командой из корня проекта."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    build_demo(args.output_dir)


if __name__ == "__main__":
    main()
