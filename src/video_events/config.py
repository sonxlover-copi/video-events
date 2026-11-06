"""Загрузка и проверка настроек офлайн-счётчика транспорта."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CounterConfig:
    """Параметры одного запуска; после чтения YAML они больше не меняются."""

    model: Path
    device: str
    classes: tuple[str, ...]
    confidence_threshold: float
    image_size: int
    tracker: str
    line_deadband_px: float
    line_start: tuple[float, float]
    line_end: tuple[float, float]


def _point(value: Any, field_name: str) -> tuple[float, float]:
    """Преобразовать точку YAML в пару координат от 0 до 1."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"'{field_name}' must contain two normalized coordinates")
    if any(
        isinstance(part, bool) or not isinstance(part, (int, float)) for part in value
    ):
        raise ValueError(f"'{field_name}' coordinates must be numbers")
    point = (float(value[0]), float(value[1]))
    if any(not math.isfinite(part) or part < 0.0 or part > 1.0 for part in point):
        raise ValueError(f"'{field_name}' coordinates must be between 0 and 1")
    return point


def load_config(path: Path) -> CounterConfig:
    """Прочитать YAML и остановиться на ошибке конфига до запуска модели."""
    config_path = path.expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read config '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in '{config_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise TypeError("Config must be a YAML mapping")

    # Явный список ключей ловит опечатки вроде `confidense_threshold` сразу.
    expected = {
        "model",
        "device",
        "classes",
        "confidence_threshold",
        "image_size",
        "tracker",
        "line_deadband_px",
        "line",
    }
    unknown = set(raw) - expected
    missing = expected - set(raw)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(sorted(missing))}")

    model_value = raw["model"]
    if not isinstance(model_value, str) or not model_value.strip():
        raise ValueError("'model' must be a non-empty local weights path")
    model = Path(model_value).expanduser()
    if not model.is_absolute():
        # Относительный путь привязан к YAML, поэтому запуск возможен из любой папки.
        model = (config_path.parent / model).resolve()
    if not model.is_file():
        raise ValueError(f"Model weights file does not exist: {model}")

    device = raw["device"]
    if not isinstance(device, str) or not device.strip():
        raise ValueError(
            "'device' must be 'auto', 'cpu', or a device index such as '0'"
        )

    classes_raw = raw["classes"]
    if not isinstance(classes_raw, list) or not classes_raw:
        raise ValueError("'classes' must be a non-empty list of model class names")
    if any(not isinstance(name, str) or not name.strip() for name in classes_raw):
        raise ValueError("Every entry in 'classes' must be a non-empty string")
    classes = tuple(name.strip() for name in classes_raw)
    if len({name.casefold() for name in classes}) != len(classes):
        raise ValueError("'classes' must not contain duplicate names")

    confidence = raw["confidence_threshold"]
    # В Python bool считается числом; явно отсекаем true/false как порог модели.
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("'confidence_threshold' must be a number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence <= 1.0:
        raise ValueError("'confidence_threshold' must be greater than 0 and at most 1")

    image_size = raw["image_size"]
    if (
        isinstance(image_size, bool)
        or not isinstance(image_size, int)
        or image_size < 32
    ):
        raise ValueError("'image_size' must be an integer of at least 32")

    tracker = raw["tracker"]
    if not isinstance(tracker, str) or not tracker.strip():
        raise ValueError("'tracker' must name an Ultralytics tracker config")
    tracker = tracker.strip()
    tracker_path = Path(tracker).expanduser()
    if tracker_path.is_absolute():
        if not tracker_path.is_file():
            raise ValueError(f"Tracker config file does not exist: {tracker_path}")
        tracker = str(tracker_path.resolve())
    else:
        # Собственный YAML трекера ищем рядом с конфигом счётчика; короткое имя
        # вроде `bytetrack.yaml` остаётся встроенным именем Ultralytics.
        local_tracker_path = (config_path.parent / tracker_path).resolve()
        if local_tracker_path.is_file():
            tracker = str(local_tracker_path)
        elif "/" in tracker or "\\" in tracker:
            raise ValueError(
                f"Tracker config file does not exist: {local_tracker_path}"
            )

    # Deadband задан в пикселях и должен быть конечным числом, иначе сторона линии
    # не сможет определиться предсказуемо.
    deadband = raw["line_deadband_px"]
    if (
        isinstance(deadband, bool)
        or not isinstance(deadband, (int, float))
        or not math.isfinite(deadband)
        or deadband < 0
    ):
        raise ValueError("'line_deadband_px' must be a non-negative number")

    line = raw["line"]
    if not isinstance(line, dict) or set(line) != {"start", "end"}:
        raise ValueError("'line' must have exactly 'start' and 'end' points")
    start = _point(line["start"], "line.start")
    end = _point(line["end"], "line.end")
    if start == end:
        raise ValueError("The line start and end points must be different")

    return CounterConfig(
        model=model,
        device=device.strip(),
        classes=classes,
        confidence_threshold=confidence,
        image_size=image_size,
        tracker=tracker.strip(),
        line_deadband_px=float(deadband),
        line_start=start,
        line_end=end,
    )
