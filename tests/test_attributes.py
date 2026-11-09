"""Проверки цвета, накопления признаков и разделения соседних треков."""

from __future__ import annotations

import unittest

from video_events.plates import PlateReading

try:
    import cv2
    import numpy as np

except ImportError:
    cv2 = None

if cv2 is not None:
    from video_events.attributes import (
        AttributeOptions,
        VehicleAttributes,
        estimate_color,
    )


class _PlateReader:
    """Отдавать заданные чтения, чтобы проверять голосование отдельно от OCR."""

    def __init__(self) -> None:
        self.readings = []
        self.calls = 0

    def read(self, image: object) -> list[PlateReading]:
        self.calls += 1
        return self.readings


@unittest.skipIf(cv2 is None, "Color tests require NumPy and OpenCV")
class VehicleAttributeTests(unittest.TestCase):
    """Проверять правила уверенности на изображениях с известным цветом."""

    def setUp(self) -> None:
        self.frame = np.full((120, 180, 3), (0, 0, 220), dtype=np.uint8)
        self.box = [0, 0, 180, 120]
        self.reader = _PlateReader()
        self.attributes = VehicleAttributes(
            AttributeOptions(colors=True, plates=True, interval=3), self.reader
        )

    def observe(self, frame_index: int, track_id: int = 1) -> dict:
        """Передать полную рамку машины без зависимости от YOLO."""
        return self.attributes.update(
            self.frame, self.box, track_id, "car", frame_index
        )

    def test_basic_colors(self) -> None:
        for name, bgr in (
            ("red", (0, 0, 220)),
            ("green", (0, 160, 0)),
            ("blue", (220, 0, 0)),
            ("yellow", (0, 220, 220)),
            ("white", (230, 230, 230)),
            ("gray", (120, 120, 120)),
            ("black", (20, 20, 20)),
            ("brown", (20, 75, 120)),
        ):
            with self.subTest(color=name):
                crop = np.full((100, 100, 3), bgr, dtype=np.uint8)
                color, support = estimate_color(crop)
                self.assertEqual(color, name)
                self.assertEqual(support, 1)

    def test_small_or_mixed_crops_are_unknown(self) -> None:
        self.assertEqual(
            estimate_color(np.zeros((10, 10, 3), dtype=np.uint8))[0], "unknown"
        )
        self.assertEqual(
            estimate_color(np.zeros((0, 0, 3), dtype=np.uint8))[0], "unknown"
        )
        # Три равные полосы внутри выбранной области кузова: доминирующего цвета нет.
        self.frame[:, :69] = (0, 0, 220)
        self.frame[:, 69:111] = (220, 0, 0)
        self.frame[:, 111:] = (0, 160, 0)
        self.assertEqual(estimate_color(self.frame)[0], "unknown")

    def test_color_needs_multiple_samples_and_ignores_frame_edges(self) -> None:
        self.frame[:35] = (0, 0, 0)
        self.assertEqual(self.observe(0)["vehicle_color"], "unknown")
        self.assertEqual(self.observe(3)["vehicle_color"], "red")

    def test_single_plate_reading_is_not_exported_as_confirmed(self) -> None:
        self.reader.readings = [PlateReading("A123BC77", 0.95, 0.9, (40, 70, 100, 90))]
        result = self.observe(0)
        self.assertEqual(result["plate_status"], "unconfirmed")
        self.assertEqual(result["plate_text"], "")
        result = self.observe(3)
        self.assertEqual(result["plate_status"], "confirmed")
        self.assertEqual(result["plate_text"], "A123BC77")
        self.assertEqual(result["plate_observations"], 2)
        self.assertEqual(result["plate_confidence"], 0.95)

    def test_sampling_interval_prevents_double_confirmation_on_adjacent_frames(
        self,
    ) -> None:
        self.reader.readings = [PlateReading("A123BC77", 0.95, 0.9, (40, 70, 100, 90))]
        self.observe(0)
        result = self.observe(1)
        self.assertEqual(result["plate_status"], "unconfirmed")
        self.assertEqual(self.reader.calls, 1)

    def test_conflicting_readings_remove_confirmation(self) -> None:
        self.reader.readings = [PlateReading("A123BC77", 0.95, 0.9, (40, 70, 100, 90))]
        self.observe(0)
        self.observe(3)
        self.reader.readings = [PlateReading("A124BC77", 0.95, 0.9, (40, 70, 100, 90))]
        self.observe(6)
        result = self.observe(9)
        self.assertEqual(result["plate_status"], "ambiguous")
        self.assertEqual(result["plate_text"], "")

    def test_multiple_plates_in_one_crop_are_not_assigned_to_a_vehicle(self) -> None:
        self.reader.readings = [
            PlateReading("A123BC77", 0.95, 0.9, (10, 20, 70, 40)),
            PlateReading("X999XX777", 0.95, 0.9, (90, 70, 160, 90)),
        ]
        self.observe(0)
        result = self.observe(3)
        self.assertEqual(result["plate_status"], "not_read")

    def test_track_states_and_final_catalog_are_independent(self) -> None:
        self.reader.readings = [PlateReading("A123BC77", 0.95, 0.9, (40, 70, 100, 90))]
        self.observe(0)
        self.observe(3)
        self.assertEqual(self.observe(4, track_id=2)["plate_text"], "")
        rows = self.attributes.vehicle_rows(10)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["plate_text"], "A123BC77")
        self.assertEqual(rows[0]["last_seen_s"], 0.3)
        self.assertEqual(rows[1]["plate_text"], "")

    def test_boxes_outside_frame_do_not_read_another_region(self) -> None:
        self.box = [-100, -100, -20, -20]
        self.observe(0)
        self.assertEqual(self.reader.calls, 0)
        self.assertEqual(self.attributes.snapshot(1)["color_samples"], 0)

    def test_disabled_attributes_do_not_invoke_ocr(self) -> None:
        attributes = VehicleAttributes(AttributeOptions(colors=True))
        result = attributes.update(self.frame, self.box, 1, "car", 0)
        self.assertEqual(result["plate_status"], "disabled")

    def test_invalid_options_are_rejected(self) -> None:
        for changes in (
            {"interval": 0},
            {"interval": True},
            {"plate_min_observations": 1},
            {"plate_min_confidence": float("nan")},
            {"plate_min_confidence": 1.1},
            {"plate_format": "other"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                AttributeOptions(**changes)
