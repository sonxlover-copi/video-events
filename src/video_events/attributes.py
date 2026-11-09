"""Оценка цвета кузова и накопление признаков внутри одного ID трекера."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from video_events.plates import PlateReader

ATTRIBUTE_FIELDS = (
    "vehicle_color",
    "color_support",
    "color_samples",
    "plate_text",
    "plate_confidence",
    "plate_support",
    "plate_observations",
    "plate_status",
)


@dataclass(frozen=True)
class AttributeOptions:
    """Настройки дополнительных проходов; старый режим остаётся доступным."""

    colors: bool = False
    plates: bool = False
    interval: int = 10
    plate_min_confidence: float = 0.85
    plate_min_observations: int = 2
    plate_format: str = "generic"

    def __post_init__(self) -> None:
        if (
            isinstance(self.interval, bool)
            or not isinstance(self.interval, int)
            or self.interval < 1
        ):
            raise ValueError("Attribute interval must be a positive integer")
        if (
            isinstance(self.plate_min_observations, bool)
            or not isinstance(self.plate_min_observations, int)
            or self.plate_min_observations < 2
        ):
            raise ValueError("A plate needs at least two matching observations")
        if (
            not math.isfinite(self.plate_min_confidence)
            or not 0 < self.plate_min_confidence <= 1
        ):
            raise ValueError("Plate confidence must be greater than 0 and at most 1")
        if self.plate_format not in {"generic", "ru"}:
            raise ValueError("Plate format must be 'generic' or 'ru'")

    @property
    def enabled(self) -> bool:
        """Показывать, нужны ли дополнительные файлы и столбцы."""
        return self.colors or self.plates


def estimate_color(crop: np.ndarray) -> tuple[str, float]:
    """Оценить основной видимый цвет; доля пикселей не является вероятностью."""
    if crop is None or crop.size == 0:
        return "unknown", 0.0
    height, width = crop.shape[:2]
    if width < 40 or height < 30:
        return "unknown", 0.0
    # Берём внутреннюю часть рамки: верх часто занят стеклом, низ — колёсами,
    # а края — фоном. Это эвристика кузова, а не сегментация автомобиля.
    body = crop[
        round(height * 0.35) : round(height * 0.8),
        round(width * 0.15) : round(width * 0.85),
    ]
    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    labels = np.full(hue.shape, "gray", dtype="<U7")
    labels[value < 55] = "black"
    labels[(saturation < 55) & (value >= 190)] = "white"
    chromatic = (saturation >= 70) & (value >= 55)
    for name, low, high in (
        ("red", 0, 10),
        ("orange", 10, 23),
        ("yellow", 23, 35),
        ("green", 35, 86),
        ("blue", 86, 131),
        ("purple", 131, 170),
        ("red", 170, 180),
    ):
        labels[chromatic & (hue >= low) & (hue < high)] = name
    labels[chromatic & (hue >= 10) & (hue < 28) & (value < 150)] = "brown"
    names, counts = np.unique(labels, return_counts=True)
    best = int(np.argmax(counts))
    support = float(counts[best] / labels.size)
    return (str(names[best]), support) if support >= 0.45 else ("unknown", support)


@dataclass
class _VehicleState:
    """Наблюдения одного трека без хранения самих кадров."""

    class_name: str
    first_frame: int
    last_frame: int
    last_sample_frame: int = -1
    color_votes: Counter[str] = field(default_factory=Counter)
    color_samples: int = 0
    plate_votes: Counter[str] = field(default_factory=Counter)
    plate_scores: Counter[str] = field(default_factory=Counter)
    plate_attempts: int = 0


class VehicleAttributes:
    """Обновлять признаки редко и подтверждать их по нескольким кадрам."""

    def __init__(
        self, options: AttributeOptions, plate_reader: PlateReader | None = None
    ) -> None:
        self.options = options
        self.reader = plate_reader
        if options.plates and self.reader is None:
            self.reader = PlateReader(
                options.plate_format, options.plate_min_confidence
            )
        self.tracks: dict[int, _VehicleState] = {}

    def update(
        self,
        frame: np.ndarray,
        box: list[float],
        track_id: int,
        class_name: str,
        frame_index: int,
    ) -> dict[str, str | int | float]:
        """Обработать рамку на исходном кадре, до нанесения любых подписей."""
        state = self.tracks.setdefault(
            track_id, _VehicleState(class_name, frame_index, frame_index)
        )
        state.last_frame = frame_index
        state.class_name = class_name
        if (
            state.last_sample_frame >= 0
            and frame_index - state.last_sample_frame < self.options.interval
        ):
            return self.snapshot(track_id)
        state.last_sample_frame = frame_index
        height, width = frame.shape[:2]
        x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
        x2, y2 = min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))
        if x2 <= x1 or y2 <= y1:
            return self.snapshot(track_id)
        crop = frame[y1:y2, x1:x2]
        if self.options.colors:
            color, _ = estimate_color(crop)
            state.color_samples += 1
            state.color_votes[color] += 1
        if (
            self.options.plates
            and self.reader is not None
            and crop.shape[1] >= 80
            and crop.shape[0] >= 40
        ):
            state.plate_attempts += 1
            readings = self.reader.read(crop)
            # В перекрывающихся рамкам могут оказаться номера соседних машин.
            # При нескольких кандидатах не выбираем один из них наугад.
            if len(readings) == 1:
                reading = readings[0]
                state.plate_votes[reading.text] += 1
                state.plate_scores[reading.text] += reading.confidence
        return self.snapshot(track_id)

    def snapshot(self, track_id: int) -> dict[str, str | int | float]:
        """Отдать только подтверждённые признаки и понятный статус номера."""
        state = self.tracks[track_id]
        color, color_support = "unknown", 0.0
        if state.color_votes:
            candidate, votes = state.color_votes.most_common(1)[0]
            color_support = votes / state.color_samples
            if candidate != "unknown" and votes >= 2 and color_support >= 0.6:
                color = candidate
        text, confidence, observations = "", 0.0, 0
        support = 0.0
        status = "not_read" if self.options.plates else "disabled"
        if state.plate_votes:
            candidate, observations = state.plate_votes.most_common(1)[0]
            support = observations / state.plate_votes.total()
            status = "unconfirmed"
            if observations >= self.options.plate_min_observations and support >= 0.6:
                text = candidate
                confidence = state.plate_scores[candidate] / observations
                status = "confirmed"
            elif len(state.plate_votes) > 1:
                status = "ambiguous"
        return {
            "vehicle_color": color,
            "color_support": round(color_support, 3),
            "color_samples": state.color_samples,
            "plate_text": text,
            "plate_confidence": round(confidence, 4),
            "plate_support": round(support, 3),
            "plate_observations": observations,
            "plate_status": status,
        }

    def vehicle_rows(self, fps: float) -> list[dict[str, Any]]:
        """Собрать итоговый каталог треков, включая машины без пересечения линии."""
        return [
            {
                "track_id": track_id,
                "class_name": state.class_name,
                "first_seen_s": round(state.first_frame / fps, 3),
                "last_seen_s": round(state.last_frame / fps, 3),
                "plate_attempts": state.plate_attempts,
                **self.snapshot(track_id),
            }
            for track_id, state in sorted(self.tracks.items())
        ]
