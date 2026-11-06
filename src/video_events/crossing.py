"""Подсчёт пересечений линии по траекториям объектов."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _TrackState:
    """Последнее положение объекта, достаточно далёкое от линии для сравнения."""

    stable_side: int | None
    stable_center: tuple[float, float] | None
    stable_frame_index: int | None
    stable_timestamp_s: float | None


class LineCrossingCounter:
    """Считать переходы между сторонами конечной линии с мёртвой зоной.

    Направление зависит от порядка концов ``start`` и ``end``. Знаковое
    расстояние вычисляется через векторное произведение и измеряется в пикселях.
    """

    def __init__(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        deadband_px: float = 8.0,
    ) -> None:
        self.start = start
        self.end = end
        self.deadband_px = deadband_px
        # Длина нужна, чтобы нормировать знаковое расстояние до линии в пиксели.
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        self._line_length = (dx * dx + dy * dy) ** 0.5
        if self._line_length == 0:
            raise ValueError("Line endpoints must be different")
        self._tracks: dict[int, _TrackState] = {}

    def _signed_distance(self, point: tuple[float, float]) -> float:
        """Вернуть расстояние до бесконечной прямой со знаком стороны."""
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return (
            dx * (point[1] - self.start[1]) - dy * (point[0] - self.start[0])
        ) / self._line_length

    def _side(self, point: tuple[float, float]) -> int | None:
        """Считать сторону только после выхода центра за deadband."""
        distance = self._signed_distance(point)
        if distance > self.deadband_px:
            return 1
        if distance < -self.deadband_px:
            return -1
        return None

    def _segment_intersection(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> tuple[float, tuple[float, float]] | None:
        """Найти пересечение движения трека с конечным отрезком линии."""
        first_distance = self._signed_distance(first)
        second_distance = self._signed_distance(second)
        denominator = first_distance - second_distance
        if denominator == 0:
            return None

        # Находим место пересечения траектории с бесконечной прямой через долю
        # пройденного пути. Затем ниже проверяем, попало ли оно в конечный отрезок.
        ratio = first_distance / denominator
        if not 0.0 <= ratio <= 1.0:
            return None
        intersection = (
            first[0] + ratio * (second[0] - first[0]),
            first[1] + ratio * (second[1] - first[1]),
        )

        line_dx = self.end[0] - self.start[0]
        line_dy = self.end[1] - self.start[1]
        line_length_squared = line_dx * line_dx + line_dy * line_dy
        # Проекция 0..1 означает, что точка лежит между концами линии, а не на её
        # продолжении за пределами кадра.
        projection = (
            (intersection[0] - self.start[0]) * line_dx
            + (intersection[1] - self.start[1]) * line_dy
        ) / line_length_squared
        return (ratio, intersection) if 0.0 <= projection <= 1.0 else None

    def update(
        self,
        *,
        track_id: int,
        class_name: str,
        confidence: float,
        center: tuple[float, float],
        frame_index: int,
        timestamp_s: float,
    ) -> dict[str, int | float | str] | None:
        """Обновить одну траекторию и вернуть событие при подтверждённом переходе."""
        state = self._tracks.get(track_id)
        if state is None:
            # Трек мог появиться уже около линии. Событие считаем только после того,
            # как у него появится стабильная исходная сторона.
            side = self._side(center)
            self._tracks[track_id] = _TrackState(
                stable_side=side,
                stable_center=center if side is not None else None,
                stable_frame_index=frame_index if side is not None else None,
                stable_timestamp_s=timestamp_s if side is not None else None,
            )
            return None

        side = self._side(center)
        if side is None:
            # Внутри deadband сохраняем последнее устойчивое положение: шум рамки
            # не должен перезаписывать сторону до завершения перехода.
            return None

        previous_side = state.stable_side
        previous_center = state.stable_center
        previous_frame_index = state.stable_frame_index
        previous_timestamp_s = state.stable_timestamp_s
        state.stable_side = side
        state.stable_center = center
        state.stable_frame_index = frame_index
        state.stable_timestamp_s = timestamp_s
        if (
            previous_side is None
            or previous_center is None
            or previous_frame_index is None
            or previous_timestamp_s is None
            or previous_side == side
        ):
            return None

        intersection = self._segment_intersection(previous_center, center)
        if intersection is None:
            return None
        ratio, crossing_point = intersection

        # Переход фиксируется лишь после выхода на новую сторону. Время и точку
        # события оцениваем между двумя устойчивыми наблюдениями трекера.
        direction = (
            "negative_to_positive" if previous_side < side else "positive_to_negative"
        )
        crossing_frame = previous_frame_index + ratio * (
            frame_index - previous_frame_index
        )
        crossing_timestamp = previous_timestamp_s + ratio * (
            timestamp_s - previous_timestamp_s
        )
        return {
            "confirmation_frame_index": frame_index,
            "crossing_frame_estimate": round(crossing_frame, 2),
            "timestamp_s": round(crossing_timestamp, 3),
            "track_id": track_id,
            "class_name": class_name,
            "confidence": round(confidence, 5),
            "direction": direction,
            "line_intersection_x_px": round(crossing_point[0], 2),
            "line_intersection_y_px": round(crossing_point[1], 2),
        }
