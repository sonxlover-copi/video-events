"""Локальное обнаружение и чтение номеров с явной подготовкой ONNX-моделей."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"


@dataclass(frozen=True)
class PlateReading:
    """Одно чтение номера; подтверждение по времени выполняется отдельно."""

    text: str
    confidence: float
    detector_confidence: float
    box: tuple[int, int, int, int]


def normalize_plate(text: str, plate_format: str = "generic") -> str:
    """Убрать разделители и проверить формат, не угадывая спорные символы."""
    # Кириллицу переводим только по визуально одинаковым буквам. Замены O→0
    # или B→8 здесь нет: такие исправления могут создать другой реальный номер.
    translation = str.maketrans("АВЕКМНОРСТУХ", "ABEKMHOPCTYX")
    normalized = re.sub(r"[\s\-]", "", text.upper().translate(translation)).rstrip("_")
    if plate_format == "ru":
        pattern = r"[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}"
    elif plate_format == "generic":
        pattern = r"[A-Z0-9]{4,12}"
    else:
        raise ValueError("Plate format must be 'generic' or 'ru'")
    return normalized if re.fullmatch(pattern, normalized, flags=re.ASCII) else ""


def _model_files() -> tuple[Path, Path, Path]:
    """Найти файлы в стандартных кешах установленных библиотек."""
    try:
        from fast_plate_ocr.inference import hub as ocr_hub
        from open_image_models.detection.core import hub as detector_hub
    except ImportError as exc:
        raise ValueError(
            "Plate recognition requires: uv sync --locked --extra plates"
        ) from exc
    detector_url = detector_hub.DETECTION_MODELS[DETECTOR_MODEL].url
    ocr_url, config_url = ocr_hub.AVAILABLE_ONNX_MODELS[OCR_MODEL]
    return (
        detector_hub.MODEL_CACHE_DIR / DETECTOR_MODEL / detector_url.rsplit("/", 1)[1],
        ocr_hub.MODEL_CACHE_DIR / OCR_MODEL / ocr_url.rsplit("/", 1)[1],
        ocr_hub.MODEL_CACHE_DIR / OCR_MODEL / config_url.rsplit("/", 1)[1],
    )


def prepare_plate_models(force: bool = False) -> dict[str, Any]:
    """Скачать веса один раз, затем проверить, что обе ONNX-сессии открываются."""
    _model_files()  # Сначала выдаём понятную ошибку, если extra не установлен.
    from fast_plate_ocr.inference.hub import download_model as download_ocr
    from open_image_models.detection.core.hub import download_model as download_detector

    download_detector(DETECTOR_MODEL, force_download=force)
    download_ocr(OCR_MODEL, force_download=force)
    reader = PlateReader()
    return reader.model_info()


class PlateReader:
    """Искать номер в BGR-изображении и читать его локально на CPU."""

    def __init__(
        self, plate_format: str = "generic", min_confidence: float = 0.85
    ) -> None:
        if plate_format not in {"generic", "ru"}:
            raise ValueError("Plate format must be 'generic' or 'ru'")
        if not math.isfinite(min_confidence) or not 0 < min_confidence <= 1:
            raise ValueError("Plate confidence must be greater than 0 and at most 1")
        self.files = _model_files()
        if any(not path.is_file() for path in self.files):
            raise ValueError(
                "Plate models are missing. Run: "
                "uv run --extra plates video-events prepare-plates"
            )
        from fast_alpr import ALPR
        from onnxruntime import SessionOptions

        # Небольшие модели быстрее и предсказуемее работают с ограниченным числом
        # CPU-потоков; CUDA остаётся у основного детектора транспорта.
        options = SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.model = ALPR(
            detector_model=DETECTOR_MODEL,
            detector_conf_thresh=0.5,
            detector_providers=["CPUExecutionProvider"],
            detector_sess_options=options,
            ocr_model=OCR_MODEL,
            ocr_device="cpu",
            ocr_providers=["CPUExecutionProvider"],
            ocr_sess_options=options,
        )
        self.plate_format = plate_format
        self.min_confidence = min_confidence

    def model_info(self) -> dict[str, Any]:
        """Сохранить имена и хеши весов, чтобы различать версии распознавания."""
        return {
            "detector": DETECTOR_MODEL,
            "ocr": OCR_MODEL,
            "provider": "CPUExecutionProvider",
            "format": self.plate_format,
            "min_confidence": self.min_confidence,
            "files": [
                {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in self.files
            ],
        }

    def read(self, image: Any) -> list[PlateReading]:
        """Вернуть пригодные чтения; мелкие и неуверенные номера отбрасываются."""
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        readings = []
        for detection in self.model.detector.predict(image):
            box = detection.bounding_box
            x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
            x2, y2 = min(width, int(box.x2)), min(height, int(box.y2))
            # Увеличение крошечной рамки не восстанавливает потерянные символы.
            if x2 - x1 < 48 or y2 - y1 < 12:
                continue
            result = self.model.ocr.predict(image[y1:y2, x1:x2])
            if result is None:
                continue
            text = normalize_plate(result.text, self.plate_format)
            scores = result.confidence
            confidence = (
                sum(scores) / len(scores)
                if isinstance(scores, list) and scores
                else float(scores)
                if not isinstance(scores, list)
                else 0.0
            )
            if (
                not text
                or not math.isfinite(confidence)
                or confidence < self.min_confidence
            ):
                continue
            readings.append(
                PlateReading(
                    text, confidence, float(detection.confidence), (x1, y1, x2, y2)
                )
            )
        return readings
