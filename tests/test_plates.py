"""Проверки нормализации номеров без загрузки моделей и обращения к сети."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from video_events.plates import PlateReader, normalize_plate


class PlateFormatTests(unittest.TestCase):
    """Отличать нормализацию записи от угадывания символов OCR."""

    def test_russian_plate_with_cyrillic_letters_and_region(self) -> None:
        self.assertEqual(normalize_plate("А 123 ВС 77", "ru"), "A123BC77")
        self.assertEqual(normalize_plate("Х999ХХ777", "ru"), "X999XX777")

    def test_wrong_russian_layout_is_rejected_without_character_substitution(
        self,
    ) -> None:
        for text in ("A12BC77", "A123BC7", "A123BC7777", "D123BC77", "A12OBC77"):
            with self.subTest(text=text):
                self.assertEqual(normalize_plate(text, "ru"), "")

    def test_generic_normalization_preserves_letters_and_digits(self) -> None:
        self.assertEqual(normalize_plate("ab-123-cd___"), "AB123CD")
        self.assertEqual(normalize_plate("O0B8"), "O0B8")

    def test_unknown_characters_and_unreasonable_lengths_are_rejected(self) -> None:
        for text in (
            "",
            "ABC",
            "A" * 13,
            "AB?123",
            "AB_1234",
            "京A12345",
            "A１２３BC77",
        ):
            with self.subTest(text=text):
                self.assertEqual(normalize_plate(text), "")

    def test_unknown_format_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_plate("AB1234", "unsupported")


class _Image:
    """Минимальный интерфейс кадра для проверки адаптера без NumPy."""

    size = 30000
    shape = (100, 200, 3)

    def __getitem__(self, key: object) -> _Image:
        return self


class PlateReaderTests(unittest.TestCase):
    """Проверять отбраковку рамок и ответов OCR без настоящих нейросетей."""

    def setUp(self) -> None:
        self.reader = PlateReader.__new__(PlateReader)
        self.reader.plate_format = "generic"
        self.reader.min_confidence = 0.85
        self.reader.model = SimpleNamespace(detector=Mock(), ocr=Mock())
        self.reader.model.detector.predict.return_value = [
            SimpleNamespace(
                bounding_box=SimpleNamespace(x1=-5, y1=20, x2=100, y2=50),
                confidence=0.9,
            )
        ]

    def test_missing_cached_models_fail_before_loading_or_downloading(self) -> None:
        with (
            patch(
                "video_events.plates._model_files",
                return_value=(Path("/missing-plate-model.onnx"),),
            ),
            self.assertRaisesRegex(ValueError, "prepare-plates"),
        ):
            PlateReader()

    def test_character_confidences_are_averaged_and_box_is_clipped(self) -> None:
        self.reader.model.ocr.predict.return_value = SimpleNamespace(
            text="AB1234", confidence=[0.9, 1.0]
        )
        result = self.reader.read(_Image())
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].confidence, 0.95)
        self.assertEqual(result[0].box, (0, 20, 100, 50))

    def test_low_empty_or_nonfinite_confidence_is_rejected(self) -> None:
        for confidence in (0.4, [], [float("nan")], [0.2, 0.9]):
            with self.subTest(confidence=confidence):
                self.reader.model.ocr.predict.return_value = SimpleNamespace(
                    text="AB1234", confidence=confidence
                )
                self.assertEqual(self.reader.read(_Image()), [])

    def test_small_plate_is_not_sent_to_ocr(self) -> None:
        self.reader.model.detector.predict.return_value[0].bounding_box.x2 = 30
        self.assertEqual(self.reader.read(_Image()), [])
        self.reader.model.ocr.predict.assert_not_called()

    def test_no_ocr_result_or_invalid_text_does_not_create_a_plate(self) -> None:
        for result in (None, SimpleNamespace(text="AB?123", confidence=0.99)):
            self.reader.model.ocr.predict.return_value = result
            self.assertEqual(self.reader.read(_Image()), [])
