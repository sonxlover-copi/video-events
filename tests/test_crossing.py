"""Проверки счётчика на коротких траекториях с заранее известным результатом."""

from __future__ import annotations

import unittest

from video_events.crossing import LineCrossingCounter


class LineCrossingCounterTests(unittest.TestCase):
    """Проверять события отдельно от качества детектора и смены ID трекера."""

    def setUp(self) -> None:
        self.counter = LineCrossingCounter((0, 0), (100, 0), deadband_px=2)

    def observe(
        self,
        center: tuple[float, float],
        frame: int,
        track_id: int = 1,
    ) -> dict[str, int | float | str] | None:
        """Передать наблюдение с фиксированными классом и частотой 10 кадров/с."""
        return self.counter.update(
            track_id=track_id,
            class_name="car",
            confidence=0.9,
            center=center,
            frame_index=frame,
            timestamp_s=frame / 10,
        )

    def test_crossing_directions_and_interpolated_time(self) -> None:
        for before, after, direction in (
            (-10, 10, "negative_to_positive"),
            (10, -10, "positive_to_negative"),
        ):
            with self.subTest(direction=direction):
                self.setUp()
                self.assertIsNone(self.observe((40, before), 0))
                event = self.observe((40, after), 4)
                self.assertIsNotNone(event)
                self.assertEqual(event["direction"], direction)
                self.assertEqual(event["timestamp_s"], 0.2)
                self.assertEqual(event["crossing_frame_estimate"], 2)
                self.assertEqual(event["confirmation_frame_index"], 4)
                self.assertEqual(event["line_intersection_x_px"], 40)
                self.assertEqual(event["line_intersection_y_px"], 0)

    def test_deadband_preserves_last_stable_position(self) -> None:
        # Касания линии и её границы не должны менять исходную сторону.
        for frame, y in enumerate((-10, -2, 1, -1, 0, 2)):
            self.assertIsNone(self.observe((40, y), frame))
        event = self.observe((40, 10), 6)
        self.assertIsNotNone(event)
        self.assertEqual(event["timestamp_s"], 0.3)
        self.assertIsNone(self.observe((40, 15), 7))

    def test_crossing_beyond_segment_is_ignored(self) -> None:
        for track_id, x in enumerate((-1, 101), start=1):
            self.assertIsNone(self.observe((x, -10), 0, track_id))
            self.assertIsNone(self.observe((x, 10), 1, track_id))

    def test_segment_endpoints_are_included(self) -> None:
        for track_id, x in enumerate((0, 100), start=1):
            self.observe((x, -10), 0, track_id)
            self.assertIsNotNone(self.observe((x, 10), 1, track_id))

    def test_track_born_on_line_needs_a_stable_starting_side(self) -> None:
        for frame, y in enumerate((0, 1, 10, 15)):
            self.assertIsNone(self.observe((40, y), frame))
        event = self.observe((40, -10), 4)
        self.assertIsNotNone(event)
        self.assertEqual(event["direction"], "positive_to_negative")

    def test_track_states_are_independent(self) -> None:
        self.observe((40, -10), 0, track_id=1)
        self.assertIsNone(self.observe((40, 10), 1, track_id=2))
        self.assertIsNotNone(self.observe((40, 10), 2, track_id=1))
        self.assertIsNotNone(self.observe((40, -10), 3, track_id=2))

    def test_return_crossing_is_a_separate_event(self) -> None:
        self.observe((40, -10), 0)
        self.assertIsNotNone(self.observe((40, 10), 1))
        self.assertIsNone(self.observe((40, 12), 2))
        self.assertIsNotNone(self.observe((40, -10), 3))

    def test_diagonal_line_and_reversed_orientation(self) -> None:
        self.counter = LineCrossingCounter((0, 0), (100, 100), deadband_px=2)
        self.observe((60, 40), 0)
        event = self.observe((40, 60), 2)
        self.assertEqual(event["direction"], "negative_to_positive")
        self.assertEqual(event["line_intersection_x_px"], 50)
        self.assertEqual(event["line_intersection_y_px"], 50)
        self.counter = LineCrossingCounter((100, 100), (0, 0), deadband_px=2)
        self.observe((60, 40), 0)
        event = self.observe((40, 60), 2)
        self.assertEqual(event["direction"], "positive_to_negative")

    def test_zero_length_line_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "endpoints"):
            LineCrossingCounter((5, 5), (5, 5))
