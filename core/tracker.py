"""
core/tracker.py
Stateful per-vehicle speed tracker.
Detects when a vehicle crosses the measurement line in either direction
and computes speed from the time it takes to traverse one car-length.
"""

from config import CAR_LENGTH_METERS, SPEED_THRESHOLD, LINE_Y_FRACTION


class SpeedTracker:
    """
    Tracks every vehicle across frames and calculates speed when it
    fully crosses the detection line.

    Usage:
        tracker = SpeedTracker(fps, frame_height)
        result  = tracker.update(track_id, x1, y1, x2, y2, frame_number)
        # result is a dict with speed info, or None if not yet measured
    """

    def __init__(self, fps: float, frame_height: int):
        self.fps          = fps
        self.line_y       = int(frame_height * LINE_Y_FRACTION)

        self._prev        = {}   # track_id -> {top, bottom}
        self._front_time  = {}   # track_id -> time front edge crossed
        self._rear_time   = {}   # track_id -> time rear edge crossed
        self._direction   = {}   # track_id -> "up" | "down"
        self._speed       = {}   # track_id -> speed_kmh
        self._measured    = set()

    @property
    def line_y_px(self):
        return self.line_y

    def get_speed(self, track_id: int):
        return self._speed.get(track_id)

    def get_direction(self, track_id: int):
        return self._direction.get(track_id)

    def is_overspeeding(self, track_id: int) -> bool:
        spd = self._speed.get(track_id)
        return spd is not None and spd >= SPEED_THRESHOLD

    def update(self, track_id: int, x1, y1, x2, y2, frame_number: int):
        """
        Call once per frame per detected vehicle.
        Returns a result dict when speed is first calculated, else None.

            result = {
                "track_id":  int,
                "speed_kmh": float,
                "direction": "up" | "down",
                "overspeed": bool,
                "frame":     int,
            }
        """
        current_time = frame_number / self.fps
        top_y, bottom_y = y1, y2

        result = None

        if track_id in self._prev:
            prev_top    = self._prev[track_id]["top"]
            prev_bottom = self._prev[track_id]["bottom"]

            mid_y      = (top_y + bottom_y) // 2
            prev_mid_y = (prev_top + prev_bottom) // 2
            moving_down = mid_y > prev_mid_y
            moving_up   = mid_y < prev_mid_y

            line_y = self.line_y

            # ── Moving DOWN (incoming) ─────────────────────
            if (
                moving_down
                and track_id not in self._front_time
                and prev_bottom <= line_y < bottom_y
            ):
                self._front_time[track_id] = current_time
                self._direction[track_id]  = "down"

            if (
                self._direction.get(track_id) == "down"
                and track_id in self._front_time
                and track_id not in self._rear_time
                and prev_top <= line_y < top_y
            ):
                self._rear_time[track_id] = current_time
                result = self._calc_speed(track_id, frame_number)

            # ── Moving UP (outgoing) ───────────────────────
            if (
                moving_up
                and track_id not in self._front_time
                and prev_top >= line_y > top_y
            ):
                self._front_time[track_id] = current_time
                self._direction[track_id]  = "up"

            if (
                self._direction.get(track_id) == "up"
                and track_id in self._front_time
                and track_id not in self._rear_time
                and prev_bottom >= line_y > bottom_y
            ):
                self._rear_time[track_id] = current_time
                result = self._calc_speed(track_id, frame_number)

        self._prev[track_id] = {"top": top_y, "bottom": bottom_y}
        return result

    def _calc_speed(self, track_id: int, frame_number: int):
        delta_t = self._rear_time[track_id] - self._front_time[track_id]
        if delta_t <= 0:
            return None

        speed_kmh = (CAR_LENGTH_METERS / delta_t) * 3.6
        self._speed[track_id] = speed_kmh
        self._measured.add(track_id)

        return {
            "track_id":  track_id,
            "speed_kmh": speed_kmh,
            "direction": self._direction[track_id],
            "overspeed": speed_kmh >= SPEED_THRESHOLD,
            "frame":     frame_number,
        }

    def reset(self):
        self._prev.clear()
        self._front_time.clear()
        self._rear_time.clear()
        self._direction.clear()
        self._speed.clear()
        self._measured.clear()
