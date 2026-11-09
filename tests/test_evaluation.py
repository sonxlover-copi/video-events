"""Проверки однозначного сопоставления событий и расчёта итоговых метрик."""

from __future__ import annotations

import unittest

from scripts.evaluate_detrac import _counts, _match_events


def event(timestamp_s: float, **changes: object) -> dict:
    """Создать событие, меняя в каждом сценарии только существенные поля."""
    return {
        "timestamp_s": timestamp_s,
        "class_name": "car",
        "direction": "negative_to_positive",
        "line_intersection_x_px": 50,
        "line_intersection_y_px": 50,
        **changes,
    }


class EventMatchingTests(unittest.TestCase):
    """Проверять неоднозначные пары и ограничения окна сопоставления."""

    def test_one_prediction_cannot_match_two_expected_events(self) -> None:
        pairs = _match_events([event(0), event(0.1)], [event(0.05)], 1, 40)
        self.assertEqual(len(pairs), 1)

    def test_matching_reassigns_pairs_to_find_both_events(self) -> None:
        # Обе эталонные записи ближе к первому предсказанию. Для двух совпадений
        # нужно переназначить уже занятую пару, а не останавливаться на ближайшей.
        pairs = _match_events([event(0), event(0.1)], [event(0.05), event(-0.9)], 1)
        self.assertEqual(len(pairs), 2)
        self.assertEqual({index for _, index in pairs}, {0, 1})

    def test_class_direction_time_and_position_restrict_matching(self) -> None:
        for prediction in (
            event(0, class_name="bus"),
            event(0, direction="positive_to_negative"),
            event(1.01),
            event(0, line_intersection_x_px=91),
        ):
            with self.subTest(prediction=prediction):
                self.assertEqual(_match_events([event(0)], [prediction], 1, 40), [])
        self.assertEqual(
            _match_events([event(0)], [event(1, line_intersection_x_px=90)], 1, 40),
            [(0, 0)],
        )

    def test_metrics_use_raw_counts_and_signed_count_error(self) -> None:
        result = _counts([event(0)] * 3, [event(0)] * 2, [(0, 0)], lambda _: True)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 1, 2))
        self.assertEqual(result["f1"], 0.4)
        self.assertEqual(result["count_error"], -1)
        self.assertEqual(result["absolute_count_error"], 1)
        self.assertEqual(result["relative_count_error"], -0.3333)

    def test_empty_groups_do_not_report_perfect_quality(self) -> None:
        for expected, predicted, f1 in (
            ([], [], None),
            ([event(0)], [], 0),
            ([], [event(0)], 0),
        ):
            with self.subTest(expected=expected, predicted=predicted):
                result = _counts(expected, predicted, [], lambda _: True)
                self.assertEqual(result["f1"], f1)
                if not expected:
                    self.assertIsNone(result["relative_count_error"])
