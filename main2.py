import argparse
import json
import os
import textwrap
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dashboard.visualization import (
    draw_boxes,
    draw_flow,
    draw_future_prediction_popup,
    draw_global_flow_indicator,
    draw_heatmap,
    draw_high_crowd_popup,
    draw_ids,
    draw_text,
)
from models.yolo_model import load_model
from processing.congestion import detect_congestion
from processing.crowd_predictor import CrowdPredictor
from processing.detection import detect_people
from processing.flow_analysis import calculate_flow
from processing.hybrid_tracker import HybridTracker
from processing.prediction_engine import PredictionSnapshot, build_prediction_snapshot, load_metrics_txt
from processing.risk_analysis import analyze_risk
from processing.tracker import SimpleTracker

WINDOW_NAME = "Crowd Present and Future Predictor"
PREVIEW_WINDOW_NAME = "Main2 Person ID Preview"
PREVIEW_SIZE = (360, 300)
BUTTON_BAR_HEIGHT = 120
DISPLAY_WIDTH = 1366
DISPLAY_HEIGHT = 768
DETECTION_INTERVAL = 3
ZOOM_PASS_INTERVAL = 3
FAST_STEP_SIZE = 2
RESULTS_DIR = "results"
CPS_RESULTS_DIR = os.path.join(RESULTS_DIR, "cps_results")
PREDICTION_METRICS_PATH = "models/crowd_predictor_metrics.txt"
PREDICTION_CSV_PATH = os.path.join(RESULTS_DIR, "prediction_output_log_main2.csv")
PREDICTION_JSONL_PATH = os.path.join(RESULTS_DIR, "prediction_output_log_main2.jsonl")
PREDICTION_EXPLAIN_PATH = os.path.join(RESULTS_DIR, "prediction_recommendations_main2.md")
PREDICTION_RESULT_TXT_LATEST = os.path.join(RESULTS_DIR, "prediction_result_main2_latest.txt")
PREDICTION_RESULT_TXT_DIR = RESULTS_DIR
PREDICTION_START_PERCENT = 10.0
FUTURE_LOOKAHEAD_SECONDS = 20.0
ALERT_MIN_CONFIDENCE_PERCENT = 60.0
ALERT_MIN_STREAK = 3
HIGH_CROWD_ON_COUNT = 18
HIGH_CROWD_OFF_COUNT = 12
CROWD_EMA_ALPHA = 0.20


@dataclass
class FrameAnalysis:
    frame_idx: int
    count: int
    future_count: int
    density_ratio: float
    elapsed_ratio: float
    elapsed_percent: float
    congestion_level: float
    risk_status: str
    flow_vectors: List[Tuple[float, float]]
    object_centers: Dict[int, Tuple[int, int]]
    snapshot: PredictionSnapshot
    recommendation: str
    explanation: str


@dataclass
class Button:
    key: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class ForecastSimulation:
    target_percent: int
    target_frame: int
    current_count: int
    predicted_count: int
    delta: int
    confidence_percent: float
    incoming_probability_percent: float
    risk_hint: str
    gate_reason: str
    action_recommendation: str
    prediction_mode: str
    incoming: bool
    incoming_threshold: int
    future_risk_status: str
    checkpoints: List[Tuple[int, int]]
    model_ready: bool


def _update_tracker_compat(tracker, boxes, frame):
    try:
        return tracker.update(boxes, frame=frame)
    except TypeError:
        return tracker.update(boxes)


def _set_tracker_mode_compat(tracker, mode):
    if hasattr(tracker, "set_crowd_mode"):
        try:
            tracker.set_crowd_mode(mode)
        except Exception:
            pass


def _compute_action_attribute_features(
    objects,
    boxes,
    previous_centers: Dict[int, Tuple[int, int]],
    frame_shape,
):
    current_centers: Dict[int, Tuple[int, int]] = {}
    displacements = []

    for x1, y1, x2, y2, obj_id in objects:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        current_centers[int(obj_id)] = (cx, cy)
        prev = previous_centers.get(int(obj_id))
        if prev is None:
            continue
        displacements.append(float(np.hypot(cx - prev[0], cy - prev[1])))

    frame_h, frame_w = frame_shape[:2]
    frame_area = max(1.0, float(frame_h * frame_w))
    frame_diag = max(1.0, float(np.hypot(frame_w, frame_h)))

    tracked_count = len(objects)
    detected_count = len(boxes)
    avg_speed_px = float(np.mean(displacements)) if displacements else 0.0
    moving_ratio = float(np.mean(np.array(displacements) > 2.0)) if displacements else 0.0
    avg_speed_norm = min(1.0, avg_speed_px / max(1.0, frame_diag * 0.08))

    track_coverage = tracked_count / max(1, detected_count)
    entry_pressure = max(0.0, (detected_count - tracked_count) / max(1, detected_count))

    box_areas = [max(0, x2 - x1) * max(0, y2 - y1) for x1, y1, x2, y2 in boxes]
    avg_box_area_norm = (float(np.mean(box_areas)) / frame_area) if box_areas else 0.0
    area_cv = 0.0
    if box_areas:
        mean_area = float(np.mean(box_areas))
        if mean_area > 1e-6:
            area_cv = float(np.std(box_areas) / mean_area)

    spread_norm = 0.0
    if current_centers:
        centers = np.array(list(current_centers.values()), dtype=np.float32)
        center_std = float(np.mean(np.std(centers, axis=0)))
        spread_norm = min(1.0, center_std / max(1.0, frame_diag * 0.35))

    occlusion_ratio = max(0.0, min(1.0, 1.0 - track_coverage))

    action_features = [
        round(avg_speed_norm, 6),
        round(moving_ratio, 6),
        round(entry_pressure, 6),
        round(min(1.0, track_coverage), 6),
    ]
    attribute_features = [
        round(avg_box_area_norm, 6),
        round(area_cv, 6),
        round(spread_norm, 6),
        round(occlusion_ratio, 6),
    ]
    return action_features, attribute_features, current_centers


class TimelineCrowdAnalyzer:
    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(CPS_RESULTS_DIR, exist_ok=True)
        cv2.setUseOptimized(True)
        cv2.setNumThreads(max(1, min(8, os.cpu_count() or 4)))
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if self.fps <= 0:
            self.fps = 25.0
        self.frame_budget_sec = 1.0 / max(1.0, self.fps)
        if self.total_frames <= 0:
            raise RuntimeError("Video has no frames.")

        self.model = load_model()
        self.predictor = CrowdPredictor(
            model_path="models/crowd_predictor.joblib",
            window_size=24,
            horizon_steps=15,
        )
        self.prediction_metrics = load_metrics_txt(PREDICTION_METRICS_PATH)

        try:
            self.tracker = HybridTracker()
        except Exception:
            self.tracker = SimpleTracker()

        self.tracker_mode = "low"
        self.smooth_detect_count = 0.0
        _set_tracker_mode_compat(self.tracker, self.tracker_mode)

        self.count_history: List[int] = []
        self.current_nowcast_history: List[int] = []
        self.future_count_history: List[int] = []
        self.density_history: List[float] = []
        self.future_risk_history: List[str] = []
        self.action_history: List[List[float]] = []
        self.attribute_history: List[List[float]] = []
        self.incoming_streak = 0
        self.previous_tracked_centers: Dict[int, Tuple[int, int]] = {}

        self.last_boxes: List[List[int]] = []
        self.last_objects: List[List[int]] = []
        self.detection_step = 0
        self._last_frame_idx = -1
        self._last_frame_image: Optional[np.ndarray] = None
        self.unique_gallery: Dict[int, Dict[str, np.ndarray]] = {}
        self.preview_sequence: List[int] = []
        self.preview_page = 0

        self.cache: Dict[int, FrameAnalysis] = {}
        self.last_analyzed_frame = -1

        self.current_frame_idx = 0
        self.running = False
        self.terminate_requested = False
        self.overlay_enabled = True
        self.last_predict_message = "Move timeline and click Predict"
        self.studio_forecast: Optional[ForecastSimulation] = None
        self.studio_candidate_forecast: Optional[ForecastSimulation] = None
        self.playback_segment_start_frame: Optional[int] = None
        self.pause_timestamps_seconds: List[float] = []
        self.pause_playback_intervals: List[Tuple[int, int]] = []
        self.session_operations: List[Dict[str, str]] = []
        self.pause_result_events: List[Dict[str, str]] = []

        self._ignore_trackbar_callback = False
        self.buttons = self._build_buttons()

        self.session_started_at = time.time()
        self.high_risk_events: List[FrameAnalysis] = []
        self.prediction_events: List[FrameAnalysis] = []
        self.prediction_log = open(PREDICTION_CSV_PATH, "w", encoding="utf-8")
        self.prediction_jsonl = open(PREDICTION_JSONL_PATH, "w", encoding="utf-8")
        self.latest_result_txt_path = PREDICTION_RESULT_TXT_LATEST
        self.prediction_log.write(
            "frame,elapsed_percent,current_count,current_nowcast,future_count,delta,confidence_percent,incoming_probability_percent,"
            "incoming,risk_status,congestion,flow_label,recommendation,gate_reason,action_recommendation,prediction_mode\n"
        )

    def _log_user_operation(self, operation: str, details: str = "") -> None:
        runtime_sec = max(0.0, time.time() - self.session_started_at)
        frame_idx = int(self.current_frame_idx)
        timeline_sec = self._seconds_from_frame(frame_idx)
        self.session_operations.append(
            {
                "runtime": self._format_time(runtime_sec),
                "frame": str(frame_idx),
                "timeline": self._format_time(timeline_sec),
                "operation": operation,
                "details": self._safe_short(details, 140),
            }
        )

    def _record_pause_event(self, reason: str = "Pause") -> None:
        self.pause_timestamps_seconds.append(self._seconds_from_frame(self.current_frame_idx))
        if self.playback_segment_start_frame is not None:
            start = max(0, int(self.playback_segment_start_frame))
            end = max(start, int(self.current_frame_idx))
            self.pause_playback_intervals.append((start, end))
            self.playback_segment_start_frame = None

        analysis = self.cache.get(self.current_frame_idx)
        if analysis is None and self.current_frame_idx >= 0:
            try:
                analysis = self.analyze_to(self.current_frame_idx)
            except Exception:
                analysis = None

        row: Dict[str, str] = {
            "reason": reason,
            "frame": str(int(self.current_frame_idx)),
            "timeline": self._format_time(self._seconds_from_frame(self.current_frame_idx)),
            "current": "NA",
            "future": "NA",
            "delta": "NA",
            "confidence": "NA",
            "risk": "NA",
            "recommendation": "NA",
        }

        if analysis is not None:
            snap = analysis.snapshot
            row.update(
                {
                    "current": str(int(snap.current_count)),
                    "future": str(int(snap.future_count)),
                    "delta": f"{int(snap.delta):+d}",
                    "confidence": f"{float(snap.confidence_percent):.1f}",
                    "risk": str(analysis.risk_status),
                    "recommendation": str(analysis.recommendation),
                }
            )

        self.pause_result_events.append(row)
        self._log_user_operation(
            "PAUSE",
            f"reason={reason}, frame={row['frame']}, timeline={row['timeline']}, now={row['current']}, future={row['future']}, delta={row['delta']}",
        )

    def _format_table_line(self, values: List[str], widths: List[int]) -> str:
        cells = []
        for idx, value in enumerate(values):
            text = (value or "").replace("\n", " ").strip()
            if len(text) > widths[idx]:
                text = text[: max(0, widths[idx] - 3)] + "..."
            cells.append(text.ljust(widths[idx]))
        return " | ".join(cells)

    def _build_operation_table_lines(self) -> List[str]:
        lines = ["User Operation Timeline", "-" * 110]
        headers = ["#", "Runtime", "Frame", "Timeline", "Operation", "Details"]
        widths = [3, 8, 7, 8, 18, 56]
        lines.append(self._format_table_line(headers, widths))
        lines.append("-" * 110)
        if not self.session_operations:
            lines.append("No user operations were recorded.")
            return lines
        for idx, op in enumerate(self.session_operations, start=1):
            lines.append(
                self._format_table_line(
                    [
                        str(idx),
                        op.get("runtime", ""),
                        op.get("frame", ""),
                        op.get("timeline", ""),
                        op.get("operation", ""),
                        op.get("details", ""),
                    ],
                    widths,
                )
            )
        return lines

    def _build_pause_result_table_lines(self) -> List[str]:
        lines = ["Pause Frames and Result at Pause", "-" * 130]
        headers = ["#", "Reason", "Frame", "Timeline", "Now", "Future", "Delta", "Conf%", "Risk", "Recommendation"]
        widths = [3, 14, 7, 8, 6, 7, 6, 6, 9, 54]
        lines.append(self._format_table_line(headers, widths))
        lines.append("-" * 130)
        if not self.pause_result_events:
            lines.append("No pause events were recorded.")
            return lines
        for idx, event in enumerate(self.pause_result_events, start=1):
            lines.append(
                self._format_table_line(
                    [
                        str(idx),
                        event.get("reason", ""),
                        event.get("frame", ""),
                        event.get("timeline", ""),
                        event.get("current", ""),
                        event.get("future", ""),
                        event.get("delta", ""),
                        event.get("confidence", ""),
                        event.get("risk", ""),
                        event.get("recommendation", ""),
                    ],
                    widths,
                )
            )
        return lines

    def _build_cps_video_analysis_summary(self, runtime_sec: float, peak_count: int, avg_count: int) -> str:
        if not self.count_history:
            return "No frames analyzed"
        return (
            f"frames={len(self.count_history)}, runtime={runtime_sec:.1f}s, peak={peak_count}, avg={avg_count}, "
            f"high_risk_events={len(self.high_risk_events)}, incoming_events={len(self.prediction_events)}"
        )

    def _write_cps_upload_report(self, runtime_sec: float, peak_count: int, avg_count: int) -> None:
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cps_path = os.path.join(CPS_RESULTS_DIR, f"cps_upload_result_{video_name}_{timestamp}.md")

        duration_seconds = self._seconds_from_frame(max(0, self.total_frames - 1))
        duration_text = self._format_time(duration_seconds)
        analysis_summary = self._build_cps_video_analysis_summary(runtime_sec, peak_count, avg_count)

        pause_count = len(self.pause_timestamps_seconds)
        pause_timestamps = ", ".join([self._format_time(sec) for sec in self.pause_timestamps_seconds]) if self.pause_timestamps_seconds else "None"
        pause_intervals = (
            "; ".join(
                [
                    f"{self._format_time(self._seconds_from_frame(start))} -> {self._format_time(self._seconds_from_frame(end))}"
                    for start, end in self.pause_playback_intervals
                ]
            )
            if self.pause_playback_intervals
            else "None"
        )

        if self.studio_forecast is not None:
            final_surge, _ = self._surge_decision(
                self.studio_forecast.gate_reason,
                self.studio_forecast.incoming_probability_percent,
                self.studio_forecast.incoming,
            )
            final_surge_text = (
                f"{final_surge} | future={self.studio_forecast.predicted_count}, delta={self.studio_forecast.delta:+d}, "
                f"confidence={self.studio_forecast.confidence_percent:.1f}%, risk={self.studio_forecast.future_risk_status}"
            )
        elif self.last_analyzed_frame >= 0 and self.last_analyzed_frame in self.cache:
            last_analysis = self.cache[self.last_analyzed_frame]
            snap = last_analysis.snapshot
            final_surge, _ = self._surge_decision(snap.gate_reason, snap.incoming_probability_percent, bool(snap.incoming))
            final_surge_text = (
                f"{final_surge} | future={snap.future_count}, delta={snap.delta:+d}, "
                f"confidence={snap.confidence_percent:.1f}%, risk={last_analysis.risk_status}"
            )
        else:
            final_surge_text = "NO_DATA"

        table_lines = [
            "# CPS Upload Result",
            "",
            f"Video: {self.video_path}",
            "",
            "| Duration Of Uploaded Video | Video Analysis | Pause Count | Pause Timestamps | Pause Playback Intervals | Final Output Surge |",
            "| --- | --- | ---: | --- | --- | --- |",
            f"| {duration_text} | {analysis_summary} | {pause_count} | {pause_timestamps} | {pause_intervals} | {final_surge_text} |",
            "",
        ]

        with open(cps_path, "w", encoding="utf-8") as f:
            f.write("\n".join(table_lines))

    def _build_buttons(self) -> List[Button]:
        y1 = DISPLAY_HEIGHT + 24
        y2 = DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT - 14
        left = 20
        gap = 12
        button_widths = {
            "run": 150,
            "stop": 150,
            "predict": 180,
            "overlay": 180,
            "reset": 190,
            "terminate": 190,
        }
        buttons: List[Button] = []
        order = ["run", "stop", "predict", "overlay", "reset", "terminate"]
        x = left
        for key in order:
            width = button_widths[key]
            buttons.append(Button(key, x, y1, x + width, y2))
            x += width + gap
        return buttons

    def _draw_timeline_progress(self, canvas: np.ndarray, analysis: FrameAnalysis) -> None:
        bar_x1 = 20
        bar_y1 = DISPLAY_HEIGHT + 4
        bar_x2 = DISPLAY_WIDTH - 20
        bar_y2 = DISPLAY_HEIGHT + 14

        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (48, 48, 48), -1)
        cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 1)

        full_w = max(1, bar_x2 - bar_x1)
        analyzed_ratio = (self.last_analyzed_frame / max(1, self.total_frames - 1))
        cursor_ratio = analysis.frame_idx / max(1, self.total_frames - 1)

        analyzed_w = int(full_w * max(0.0, min(1.0, analyzed_ratio)))
        cursor_x = bar_x1 + int(full_w * max(0.0, min(1.0, cursor_ratio)))
        if analyzed_w > 0:
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + analyzed_w, bar_y2), (23, 103, 163), -1)
        cv2.line(canvas, (cursor_x, bar_y1 - 2), (cursor_x, bar_y2 + 2), (0, 220, 255), 2)

        frame_sec = self._seconds_from_frame(analysis.frame_idx)
        total_sec = self._seconds_from_frame(self.total_frames - 1)
        cv2.putText(
            canvas,
            f"Timeline: {self._format_time(frame_sec)} / {self._format_time(total_sec)}  |  Frame {analysis.frame_idx + 1}/{self.total_frames}",
            (20, DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (230, 230, 230),
            1,
        )

    def _button_palette(self, key: str, active: bool) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        active_fill = {
            "run": (28, 181, 103),
            "stop": (38, 38, 220),
            "predict": (14, 118, 248),
            "overlay": (0, 155, 185),
            "reset": (0, 165, 255),
            "terminate": (60, 60, 205),
        }
        if active:
            return active_fill.get(key, (0, 180, 255)), (12, 12, 12)
        return (35, 35, 35), (245, 245, 245)

    def _safe_short(self, text: str, max_chars: int) -> str:
        cleaned = " ".join((text or "").strip().split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max(0, max_chars - 3)] + "..."

    def _get_frame_for_render(self, frame_idx: int) -> np.ndarray:
        if self._last_frame_idx == frame_idx and self._last_frame_image is not None:
            return self._last_frame_image.copy()
        return self._read_frame(frame_idx)

    def _surge_decision(self, gate_reason: str, incoming_probability_percent: float, incoming: bool) -> Tuple[str, str]:
        if gate_reason == "PASS" and (incoming or incoming_probability_percent >= 60.0):
            return "YES", "Gate pass with strong incoming probability"
        if gate_reason == "PASS" and incoming_probability_percent >= 50.0:
            return "YES", "Gate pass with moderate incoming probability"
        if gate_reason == "DELTA_BELOW_THRESHOLD":
            return "NO", "Predicted increase below alert threshold"
        if gate_reason == "CONFIDENCE_TOO_LOW":
            return "NO", "Confidence below minimum threshold"
        if gate_reason == "STREAK_NOT_MET":
            return "NO", "Trend streak not strong enough yet"
        return "NO", "Model gate did not pass surge criteria"

    def _update_unique_gallery(self, frame: np.ndarray) -> None:
        new_ids: List[int] = []
        for x1, y1, x2, y2, person_id in self.last_objects:
            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1], x2))
            y2 = max(0, min(frame.shape[0], y2))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pid = int(person_id)
            area = int((x2 - x1) * (y2 - y1))
            existing = self.unique_gallery.get(pid)
            if existing is None:
                self.unique_gallery[pid] = {"image": crop.copy(), "area": area}
                new_ids.append(pid)
            elif area > int(existing.get("area", 0)):
                self.unique_gallery[pid] = {"image": crop.copy(), "area": area}

        if new_ids:
            self.preview_sequence.extend(new_ids)

        if self.preview_sequence and len(self.preview_sequence) > 4:
            total_pages = max(1, (len(self.preview_sequence) + 3) // 4)
            self.preview_page = (self.preview_page + 1) % total_pages

    def _build_preview_frame(self) -> np.ndarray:
        preview_w, preview_h = PREVIEW_SIZE
        canvas = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        cv2.rectangle(canvas, (0, 0), (preview_w - 1, preview_h - 1), (0, 255, 255), 2)
        cv2.putText(canvas, f"Unique IDs: {len(self.unique_gallery)}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        if not self.preview_sequence:
            cv2.putText(canvas, "No tracked IDs yet", (82, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 1)
            return canvas

        cards_per_page = 4
        total_pages = max(1, (len(self.preview_sequence) + cards_per_page - 1) // cards_per_page)
        self.preview_page = min(self.preview_page, total_pages - 1)

        start = self.preview_page * cards_per_page
        page_ids = self.preview_sequence[start:start + cards_per_page]

        card_w, card_h = 160, 116
        x_positions = [10, 190]
        y_positions = [40, 164]

        for idx, pid in enumerate(page_ids):
            row = idx // 2
            col = idx % 2
            x = x_positions[col]
            y = y_positions[row]
            card = np.full((card_h, card_w, 3), 30, dtype=np.uint8)

            entry = self.unique_gallery.get(pid)
            if entry is not None:
                image = entry["image"]
                avail_w = card_w - 10
                avail_h = card_h - 28
                scale = min(avail_w / max(1, image.shape[1]), avail_h / max(1, image.shape[0]))
                rw = max(1, int(image.shape[1] * scale))
                rh = max(1, int(image.shape[0] * scale))
                resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
                ox = (card_w - rw) // 2
                oy = 4 + (avail_h - rh) // 2
                card[oy:oy + rh, ox:ox + rw] = resized

            cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (0, 255, 255), 1)
            cv2.rectangle(card, (0, card_h - 24), (card_w - 1, card_h - 1), (40, 40, 40), -1)
            cv2.putText(card, f"ID: {pid}", (8, card_h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
            canvas[y:y + card_h, x:x + card_w] = card

        cv2.putText(canvas, f"Page {self.preview_page + 1}/{total_pages}", (230, preview_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
        return canvas

    def _confidence_band(self, confidence_percent: float) -> Tuple[str, Tuple[int, int, int]]:
        if confidence_percent >= 75.0:
            return "HIGH", (120, 255, 170)
        if confidence_percent >= 60.0:
            return "MEDIUM", (0, 220, 255)
        return "LOW", (0, 165, 255)

    def _predict_current_nowcast(self) -> int:
        if not self.count_history:
            return 0
        window = self.count_history[-12:]
        if len(window) < 3:
            return int(window[-1])
        values = np.array(window, dtype=np.float32)
        weights = np.linspace(0.6, 1.4, num=len(values), dtype=np.float32)
        baseline = float(np.average(values, weights=weights))
        velocity = float(values[-1] - values[max(0, len(values) - 4)])
        nowcast = baseline + (0.18 * velocity)
        return max(0, int(round(nowcast)))

    def _future_horizon_frames(self) -> int:
        return max(1, int(round(self.fps * FUTURE_LOOKAHEAD_SECONDS)))

    def _predict_future_seconds_ahead(
        self,
        elapsed_ratio: float,
        action_features: List[float],
        attribute_features: List[float],
        fallback_count: int,
    ) -> int:
        if not (self.predictor.is_trained() and self.count_history and self.density_history):
            return int(fallback_count)

        horizon_frames = self._future_horizon_frames()
        simulated_counts = list(self.count_history)
        simulated_densities = list(self.density_history)
        fallback_density = float(simulated_densities[-1])
        frame_ratio_step = 1.0 / max(1.0, float(self.total_frames - 1))

        for step in range(horizon_frames):
            next_elapsed_ratio = min(1.0, float(elapsed_ratio) + ((step + 1) * frame_ratio_step))
            next_count = self.predictor.predict(
                simulated_counts,
                simulated_densities,
                next_elapsed_ratio,
                action_features=action_features,
                attribute_features=attribute_features,
            )
            simulated_counts.append(max(0, int(next_count)))
            simulated_densities.append(fallback_density)

        return int(simulated_counts[-1]) if simulated_counts else int(fallback_count)

    def _format_future_horizon(self, from_frame: int, to_frame: int) -> str:
        delta_frames = max(0, int(to_frame) - int(from_frame))
        sec = delta_frames / max(1.0, self.fps)
        return f"{delta_frames} frames (~{sec:.2f}s)"

    def _simulate_forecast_to_percent(self, analysis: FrameAnalysis, target_percent: int) -> ForecastSimulation:
        target_percent = int(max(11, min(100, target_percent)))
        target_frame = int(round((target_percent / 100.0) * max(1, self.total_frames - 1)))
        target_frame = max(analysis.frame_idx, min(self.total_frames - 1, target_frame))

        model_ready = bool(
            self.predictor.is_trained()
            and len(self.count_history) >= self.predictor.window_size
            and len(self.density_history) >= self.predictor.window_size
        )

        simulated_counts = list(self.count_history)
        simulated_densities = list(self.density_history)
        simulated_actions = list(self.action_history)
        simulated_attributes = list(self.attribute_history)

        fallback_density = simulated_densities[-1] if simulated_densities else float(analysis.density_ratio)
        fallback_action = simulated_actions[-1] if simulated_actions else [0.0, 0.0, 0.0, 0.0]
        fallback_attribute = simulated_attributes[-1] if simulated_attributes else [0.0, 0.0, 0.0, 0.0]

        current_sim_frame = analysis.frame_idx
        while current_sim_frame < target_frame:
            next_elapsed_ratio = (current_sim_frame + 1) / max(1, self.total_frames - 1)

            if model_ready:
                nn_pred = self.predictor.predict(
                    simulated_counts,
                    simulated_densities,
                    next_elapsed_ratio,
                    action_features=fallback_action,
                    attribute_features=fallback_attribute,
                )
                next_count = max(0, int(nn_pred))
            else:
                next_count = int(simulated_counts[-1] if simulated_counts else analysis.count)

            simulated_counts.append(next_count)
            simulated_densities.append(float(fallback_density))
            simulated_actions.append(list(fallback_action))
            simulated_attributes.append(list(fallback_attribute))
            current_sim_frame += 1

        predicted_count = int(simulated_counts[target_frame]) if target_frame < len(simulated_counts) else int(analysis.count)

        snapshot = build_prediction_snapshot(
            current_count=int(analysis.count),
            nn_pred=predicted_count,
            count_history=simulated_counts[: target_frame + 1],
            metrics=self.prediction_metrics,
            prediction_active=analysis.elapsed_percent >= PREDICTION_START_PERCENT,
            nn_ready=model_ready,
            incoming_streak=self.incoming_streak,
            min_confidence_percent=ALERT_MIN_CONFIDENCE_PERCENT,
            min_streak=ALERT_MIN_STREAK,
        )

        percent_marks = [0.0, 0.25, 0.50, 0.75, 1.0]
        checkpoints: List[Tuple[int, int]] = []
        start_frame = analysis.frame_idx
        span = max(1, target_frame - start_frame)
        for mark in percent_marks:
            frame = start_frame + int(round(span * mark))
            frame = max(start_frame, min(target_frame, frame))
            value = int(simulated_counts[frame]) if frame < len(simulated_counts) else predicted_count
            checkpoints.append((frame, value))

        return ForecastSimulation(
            target_percent=target_percent,
            target_frame=target_frame,
            current_count=int(analysis.count),
            predicted_count=int(predicted_count),
            delta=int(predicted_count - int(analysis.count)),
            confidence_percent=float(snapshot.confidence_percent),
            incoming_probability_percent=float(snapshot.incoming_probability_percent),
            risk_hint=str(snapshot.risk_hint),
            gate_reason=str(snapshot.gate_reason),
            action_recommendation=str(snapshot.action_recommendation),
            prediction_mode=str(snapshot.prediction_mode),
            incoming=bool(snapshot.incoming),
            incoming_threshold=int(snapshot.incoming_threshold),
            future_risk_status=analyze_risk(int(analysis.count), int(predicted_count), float(analysis.congestion_level)),
            checkpoints=checkpoints,
            model_ready=model_ready,
        )

    def _open_prediction_timeline_window(self, analysis: FrameAnalysis) -> None:
        window = "Prediction Timeline Studio"
        slider_name = "Future %"
        if analysis.elapsed_percent < PREDICTION_START_PERCENT:
            self.last_predict_message = (
                f"Prediction starts after {PREDICTION_START_PERCENT:.0f}% analysis. "
                f"Current progress is {analysis.elapsed_percent:.1f}%."
            )
            self.render(force_predict=False)
            return

        initial = int(max(11, min(100, round(max(analysis.elapsed_percent + 1.0, 11.0)))))
        selected_percent = initial
        applied_message: Optional[str] = None
        begin_requested = False
        panel_h, panel_w = 600, 1080
        begin_btn = (690, 516, 1030, 570)

        def _on_change(val: int) -> None:
            nonlocal selected_percent
            selected_percent = max(11, min(100, int(val)))

        def _on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
            nonlocal begin_requested
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            x1, y1, x2, y2 = begin_btn
            if x1 <= x <= x2 and y1 <= y <= y2:
                begin_requested = True

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, panel_w, panel_h)
        cv2.setMouseCallback(window, _on_mouse)
        cv2.createTrackbar(slider_name, window, initial, 100, _on_change)

        try:
            while True:
                simulation = self._simulate_forecast_to_percent(analysis, selected_percent)
                current_nowcast = self._predict_current_nowcast()
                horizon_text = self._format_future_horizon(analysis.frame_idx, simulation.target_frame)
                future_seconds = max(0.0, (simulation.target_frame - analysis.frame_idx) / max(1.0, self.fps))
                self.studio_candidate_forecast = simulation
                surge_label, surge_reason = self._surge_decision(
                    simulation.gate_reason,
                    simulation.incoming_probability_percent,
                    simulation.incoming,
                )
                confidence_band, confidence_color = self._confidence_band(simulation.confidence_percent)

                panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                panel[:] = (18, 22, 30)

                cv2.putText(panel, "PREDICTION TIMELINE STUDIO", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 220, 255), 2)
                cv2.putText(
                    panel,
                    "Model output uses trained crowd predictor + observed current-video history. No synthetic metadata is injected.",
                    (24, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (235, 235, 235),
                    1,
                )
                cv2.line(panel, (24, 84), (1056, 84), (70, 70, 70), 1)

                bar_x1, bar_y1, bar_x2, bar_y2 = 36, 124, 1044, 152
                cv2.rectangle(panel, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
                cv2.rectangle(panel, (bar_x1, bar_y1), (bar_x2, bar_y2), (100, 100, 100), 1)
                width = bar_x2 - bar_x1
                current_x = bar_x1 + int(width * (max(0.0, min(1.0, analysis.elapsed_percent / 100.0))))
                target_x = bar_x1 + int(width * (simulation.target_percent / 100.0))
                cv2.rectangle(panel, (bar_x1, bar_y1), (current_x, bar_y2), (20, 115, 180), -1)
                cv2.rectangle(panel, (current_x, bar_y1), (target_x, bar_y2), (70, 145, 70), -1)
                cv2.line(panel, (target_x, bar_y1 - 6), (target_x, bar_y2 + 6), (0, 220, 255), 2)

                cv2.putText(panel, f"Current timeline: {analysis.elapsed_percent:.1f}%", (36, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
                cv2.putText(panel, f"Target timeline: {simulation.target_percent}%  (frame {simulation.target_frame + 1}/{self.total_frames})", (565, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1)
                cv2.putText(panel, f"Future horizon: {horizon_text}", (36, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1)
                cv2.putText(panel, f"Predicting next: {future_seconds:.2f} seconds", (565, 174), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1)

                cv2.putText(panel, f"Now Crowd: {analysis.count}", (40, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (250, 250, 250), 2)
                cv2.putText(panel, f"Current Nowcast: {current_nowcast}", (420, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)

                if begin_requested:
                    cv2.putText(panel, f"Predicted Crowd: {simulation.predicted_count}", (40, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 255, 255), 2)
                    cv2.putText(panel, f"Delta: {simulation.delta:+d}", (40, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (110, 255, 140), 2)
                    cv2.putText(panel, f"Confidence: {simulation.confidence_percent:.1f}% ({confidence_band})", (420, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.70, confidence_color, 2)
                    cv2.putText(panel, f"Incoming Prob: {simulation.incoming_probability_percent:.1f}%", (420, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)
                    cv2.putText(panel, f"Future Crowd Surge: {surge_label}", (420, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (120, 255, 170) if surge_label == "YES" else (210, 210, 210), 2)
                    cv2.putText(panel, f"Future Risk: {simulation.future_risk_status}", (420, 364), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)
                    cv2.putText(panel, "Press Enter to apply this prediction to main screen", (420, 396), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1)
                else:
                    cv2.putText(panel, "Predicted Crowd: waiting for begin", (40, 282), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (205, 205, 205), 2)
                    cv2.putText(panel, "Delta: --", (40, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.76, (185, 185, 185), 2)
                    cv2.putText(panel, "Click Begin Prediction to generate result", (420, 302), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

                cp_text = " | ".join([f"{int((frame / max(1, self.total_frames - 1)) * 100)}%:{count}" for frame, count in simulation.checkpoints])
                cv2.putText(panel, self._safe_short(f"Simulation checkpoints -> {cp_text}", 146), (40, 444), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

                gate_ok = "MET" if simulation.gate_reason == "PASS" else "NOT_MET"
                threshold_ok = "MET" if simulation.delta >= simulation.incoming_threshold else "NOT_MET"
                cv2.putText(panel, "How prediction computed", (40, 478), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 220, 255), 2)
                cv2.putText(
                    panel,
                    f"Window={self.predictor.window_size} Horizon={self.predictor.horizon_steps} | AnalyzedFrames={len(self.count_history)} | ModelReady={int(simulation.model_ready)}",
                    (40, 504),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                )
                cv2.putText(
                    panel,
                    f"Gate={simulation.gate_reason} ({gate_ok}) | Threshold={simulation.incoming_threshold} ({threshold_ok}) | PredictionMode={simulation.prediction_mode}",
                    (40, 528),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (220, 220, 220),
                    1,
                )

                btn_color = (26, 164, 98) if not begin_requested else (12, 132, 80)
                x1, y1, x2, y2 = begin_btn
                cv2.rectangle(panel, (x1, y1), (x2, y2), btn_color, -1)
                cv2.rectangle(panel, (x1, y1), (x2, y2), (230, 230, 230), 2)
                cv2.putText(panel, "BEGIN PREDICTION", (x1 + 38, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (10, 10, 10), 2)
                cv2.putText(panel, "Esc/Q: Close", (40, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (210, 210, 210), 1)

                cv2.imshow(window, panel)
                key = cv2.waitKey(35) & 0xFF
                if key in (27, ord("q")):
                    break
                if key in (13, ord("p")) and begin_requested:
                    self.studio_forecast = simulation
                    self.running = False
                    # Keep current timeline position to avoid expensive re-analysis jump on apply.
                    # The selected studio forecast is still shown immediately in the main overlay.
                    surge_label, _ = self._surge_decision(
                        simulation.gate_reason,
                        simulation.incoming_probability_percent,
                        simulation.incoming,
                    )
                    applied_message = (
                        f"Applied studio forecast @ {simulation.target_percent}% (frame {simulation.target_frame + 1}): "
                        f"now {analysis.count} -> future {simulation.predicted_count} (confidence {simulation.confidence_percent:.1f}%, risk {simulation.future_risk_status}, surge {surge_label})"
                    )
                    self._log_user_operation(
                        "APPLY_STUDIO_FORECAST",
                        f"target={simulation.target_percent}%, frame={simulation.target_frame + 1}, future={simulation.predicted_count}, delta={simulation.delta:+d}, confidence={simulation.confidence_percent:.1f}%",
                    )
                    break
        finally:
            cv2.destroyWindow(window)

        if applied_message:
            self.last_predict_message = applied_message
            self.render(force_predict=False)

    def _button_label(self, key: str) -> str:
        if key == "run":
            return "Run"
        if key == "stop":
            return "Stop"
        if key == "predict":
            return "Predict Timeline"
        if key == "overlay":
            return f"Overlay: {'ON' if self.overlay_enabled else 'OFF'}"
        if key == "reset":
            return "Reboot Timeline"
        if key == "terminate":
            return "End Prediction"
        return key

    def _flow_label(self, flow_vectors: List[Tuple[float, float]]) -> str:
        if not flow_vectors:
            return "STABLE"
        avg_dx = sum(v[0] for v in flow_vectors) / len(flow_vectors)
        avg_dy = sum(v[1] for v in flow_vectors) / len(flow_vectors)
        if abs(avg_dx) < 1 and abs(avg_dy) < 1:
            return "STABLE"
        if abs(avg_dx) >= abs(avg_dy):
            return "RIGHT" if avg_dx > 0 else "LEFT"
        return "DOWN" if avg_dy > 0 else "UP"

    def _make_recommendation(
        self,
        snapshot: PredictionSnapshot,
        risk_status: str,
        congestion_level: float,
        flow_label: str,
    ) -> Tuple[str, str]:
        rec = "NORMAL_MONITORING"
        if risk_status == "CRITICAL" or snapshot.action_recommendation == "INTERVENE":
            rec = "IMMEDIATE_INTERVENTION"
        elif risk_status in {"HIGH", "SURGE"} or snapshot.incoming:
            rec = "PREPARE_MARSHAL_TEAM"
        elif snapshot.incoming_raw:
            rec = "WATCH_ENTRY_POINTS"

        explanation = (
            f"Gate={snapshot.gate_reason}, trend={snapshot.risk_hint}, flow={flow_label}, "
            f"congestion={congestion_level:.1f}, action={snapshot.action_recommendation}"
        )
        return rec, explanation

    def _seconds_from_frame(self, frame_idx: int) -> float:
        if self.fps <= 0:
            return 0.0
        return float(frame_idx) / self.fps

    def _format_time(self, seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        total = int(seconds)
        hh = total // 3600
        mm = (total % 3600) // 60
        ss = total % 60
        return f"{hh:02d}:{mm:02d}:{ss:02d}"

    def _read_frame(self, frame_idx: int) -> np.ndarray:
        frame_idx = max(0, min(self.total_frames - 1, frame_idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame {frame_idx}.")
        resized = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)
        self._last_frame_idx = frame_idx
        self._last_frame_image = resized.copy()
        return resized

    def _reset_analysis_state(self) -> None:
        self.cache.clear()
        self.count_history.clear()
        self.current_nowcast_history.clear()
        self.future_count_history.clear()
        self.density_history.clear()
        self.future_risk_history.clear()
        self.action_history.clear()
        self.attribute_history.clear()
        self.incoming_streak = 0
        self.previous_tracked_centers.clear()
        self.last_boxes = []
        self.last_objects = []
        self.last_analyzed_frame = -1
        self.detection_step = 0
        self.smooth_detect_count = 0.0
        self._last_frame_idx = -1
        self._last_frame_image = None
        self.unique_gallery.clear()
        self.preview_sequence.clear()
        self.preview_page = 0
        self.tracker_mode = "low"
        try:
            self.tracker = HybridTracker()
        except Exception:
            self.tracker = SimpleTracker()
        _set_tracker_mode_compat(self.tracker, self.tracker_mode)

    def _analyze_frame(self, frame_idx: int) -> FrameAnalysis:
        frame_shape = (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3)
        frame_for_detection: Optional[np.ndarray] = None

        if frame_idx % DETECTION_INTERVAL == 0:
            frame_for_detection = self._read_frame(frame_idx)
            frame_shape = frame_for_detection.shape
            self.detection_step += 1
            use_zoom = (self.detection_step % ZOOM_PASS_INTERVAL == 0)
            boxes = detect_people(self.model, frame_for_detection, use_zoom=use_zoom)

            raw_detect_count = len(boxes)
            if frame_idx <= DETECTION_INTERVAL:
                self.smooth_detect_count = float(raw_detect_count)
            else:
                self.smooth_detect_count = (
                    CROWD_EMA_ALPHA * float(raw_detect_count)
                    + (1.0 - CROWD_EMA_ALPHA) * self.smooth_detect_count
                )

            if self.tracker_mode == "low" and self.smooth_detect_count >= HIGH_CROWD_ON_COUNT:
                self.tracker_mode = "high"
                _set_tracker_mode_compat(self.tracker, self.tracker_mode)
            elif self.tracker_mode == "high" and self.smooth_detect_count <= HIGH_CROWD_OFF_COUNT:
                self.tracker_mode = "low"
                _set_tracker_mode_compat(self.tracker, self.tracker_mode)

            objects = _update_tracker_compat(self.tracker, boxes, frame_for_detection)
            self.last_boxes = boxes
            self.last_objects = objects
        else:
            boxes = self.last_boxes
            objects = self.last_objects

        object_centers = {obj_id: ((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, obj_id in objects}

        tracked_count = len(objects)
        detected_count = len(boxes)
        coverage = tracked_count / max(1, detected_count)
        if detected_count == 0:
            fused_count = tracked_count
        elif coverage < 0.70:
            fused_count = max(tracked_count, detected_count)
        else:
            fused_count = max(
                tracked_count,
                int(round(0.70 * tracked_count + 0.30 * detected_count)),
            )
        self.count_history.append(int(fused_count))

        frame_area = max(1, frame_shape[0] * frame_shape[1])
        box_area_sum = 0
        for x1, y1, x2, y2 in boxes:
            box_area_sum += max(0, x2 - x1) * max(0, y2 - y1)
        density_ratio = min(1.0, box_area_sum / frame_area)
        self.density_history.append(float(density_ratio))

        action_features, attribute_features, self.previous_tracked_centers = _compute_action_attribute_features(
            objects,
            boxes,
            self.previous_tracked_centers,
            frame_shape,
        )
        self.action_history.append(action_features)
        self.attribute_history.append(attribute_features)

        recent_window = self.count_history[-10:]
        count = int(round(sum(recent_window) / len(recent_window)))

        elapsed_ratio = frame_idx / max(1, self.total_frames - 1)
        elapsed_percent = elapsed_ratio * 100.0
        prediction_active = elapsed_percent >= PREDICTION_START_PERCENT
        nn_ready = prediction_active and self.predictor.is_trained() and len(self.count_history) >= self.predictor.window_size

        if nn_ready:
            nn_pred = self._predict_future_seconds_ahead(
                elapsed_ratio=elapsed_ratio,
                action_features=action_features,
                attribute_features=attribute_features,
                fallback_count=count,
            )
        else:
            nn_pred = count

        pre_snapshot = build_prediction_snapshot(
            current_count=count,
            nn_pred=nn_pred,
            count_history=self.count_history,
            metrics=self.prediction_metrics,
            prediction_active=prediction_active,
            nn_ready=nn_ready,
            incoming_streak=self.incoming_streak,
            min_confidence_percent=ALERT_MIN_CONFIDENCE_PERCENT,
            min_streak=ALERT_MIN_STREAK,
        )

        if pre_snapshot.incoming_raw:
            self.incoming_streak += 1
        else:
            self.incoming_streak = 0

        snapshot = build_prediction_snapshot(
            current_count=count,
            nn_pred=nn_pred,
            count_history=self.count_history,
            metrics=self.prediction_metrics,
            prediction_active=prediction_active,
            nn_ready=nn_ready,
            incoming_streak=self.incoming_streak,
            min_confidence_percent=ALERT_MIN_CONFIDENCE_PERCENT,
            min_streak=ALERT_MIN_STREAK,
        )
        future_count = snapshot.future_count if prediction_active else count
        current_nowcast = self._predict_current_nowcast()
        self.current_nowcast_history.append(int(current_nowcast))

        congestion_level = detect_congestion(boxes, frame_shape)
        risk_status = analyze_risk(count, future_count, congestion_level)
        self.future_count_history.append(int(future_count))
        self.future_risk_history.append(str(risk_status))
        flow_vectors = calculate_flow(self.tracker)
        flow_label = self._flow_label(flow_vectors)
        recommendation, explanation = self._make_recommendation(snapshot, risk_status, congestion_level, flow_label)

        analysis = FrameAnalysis(
            frame_idx=frame_idx,
            count=count,
            future_count=future_count,
            density_ratio=float(density_ratio),
            elapsed_ratio=float(elapsed_ratio),
            elapsed_percent=float(elapsed_percent),
            congestion_level=float(congestion_level),
            risk_status=risk_status,
            flow_vectors=flow_vectors,
            object_centers=object_centers,
            snapshot=snapshot,
            recommendation=recommendation,
            explanation=explanation,
        )

        self.prediction_log.write(
            f"{frame_idx},{elapsed_percent:.3f},{snapshot.current_count},{current_nowcast},{snapshot.future_count},{snapshot.delta},"
            f"{snapshot.confidence_percent:.1f},{snapshot.incoming_probability_percent:.1f},{int(snapshot.incoming)},"
            f"{risk_status},{congestion_level:.3f},{flow_label},{recommendation},{snapshot.gate_reason},"
            f"{snapshot.action_recommendation},{snapshot.prediction_mode}\n"
        )
        self.prediction_jsonl.write(
            json.dumps(
                {
                    "frame": frame_idx,
                    "elapsed_percent": round(elapsed_percent, 3),
                    "current_count": snapshot.current_count,
                    "current_nowcast": int(current_nowcast),
                    "future_count": snapshot.future_count,
                    "delta": snapshot.delta,
                    "confidence_percent": snapshot.confidence_percent,
                    "incoming_probability_percent": snapshot.incoming_probability_percent,
                    "incoming": bool(snapshot.incoming),
                    "risk_status": risk_status,
                    "congestion_level": round(float(congestion_level), 3),
                    "flow_label": flow_label,
                    "recommendation": recommendation,
                    "explanation": explanation,
                    "gate_reason": snapshot.gate_reason,
                    "action_recommendation": snapshot.action_recommendation,
                }
            )
            + "\n"
        )

        if risk_status in {"HIGH", "CRITICAL", "SURGE"}:
            self.high_risk_events.append(analysis)
        if snapshot.incoming or snapshot.incoming_raw:
            self.prediction_events.append(analysis)

        return analysis

    def analyze_to(self, target_frame: int) -> FrameAnalysis:
        target_frame = max(0, min(self.total_frames - 1, target_frame))

        if target_frame in self.cache:
            return self.cache[target_frame]

        if target_frame <= self.last_analyzed_frame:
            self._reset_analysis_state()

        start = self.last_analyzed_frame + 1
        if start < 0:
            start = 0

        latest: Optional[FrameAnalysis] = None
        for idx in range(start, target_frame + 1):
            latest = self._analyze_frame(idx)
            self.cache[idx] = latest
            self.last_analyzed_frame = idx

        if latest is None:
            latest = self._analyze_frame(target_frame)
            self.cache[target_frame] = latest
            self.last_analyzed_frame = target_frame

        return latest

    def _draw_buttons(self, canvas: np.ndarray) -> None:
        analysis = self.cache.get(self.current_frame_idx)
        if analysis is None:
            analysis = self.cache.get(self.last_analyzed_frame)
        if analysis is None:
            analysis = self.analyze_to(self.current_frame_idx)
        self._draw_timeline_progress(canvas, analysis)

        for button in self.buttons:
            active = False
            if button.key == "run" and self.running:
                active = True
            if button.key == "stop" and not self.running:
                active = True
            if button.key == "overlay" and self.overlay_enabled:
                active = True

            fill, text_col = self._button_palette(button.key, active)
            cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), fill, -1)
            cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), (235, 235, 235), 2)
            cv2.putText(
                canvas,
                self._button_label(button.key),
                (button.x1 + 10, button.y1 + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                text_col,
                2,
            )

        cv2.putText(
            canvas,
            "Keys: r=Run  s=Stop  p=Predict Timeline  o=Overlay  c=Reboot  x=End Prediction  q=Quit",
            (20, DISPLAY_HEIGHT + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
        )

    def _overlay_custom_panels(self, frame: np.ndarray, analysis: FrameAnalysis) -> None:
        snap = analysis.snapshot
        left_x1, left_y1, left_x2, left_y2 = 16, 14, 760, 198
        right_x1, right_y1, right_x2, right_y2 = 820, 14, DISPLAY_WIDTH - 16, 248

        overlay = frame.copy()
        cv2.rectangle(overlay, (left_x1, left_y1), (left_x2, left_y2), (12, 18, 25), -1)
        cv2.rectangle(overlay, (right_x1, right_y1), (right_x2, right_y2), (28, 18, 45), -1)
        cv2.addWeighted(overlay, 0.52, frame, 0.48, 0, frame)
        cv2.rectangle(frame, (left_x1, left_y1), (left_x2, left_y2), (0, 220, 255), 2)
        cv2.rectangle(frame, (right_x1, right_y1), (right_x2, right_y2), (0, 220, 255), 2)

        current_nowcast = self._predict_current_nowcast()
        nowcast_delta = int(current_nowcast - int(snap.current_count))
        state_label = "ACTIVE" if (analysis.elapsed_percent >= PREDICTION_START_PERCENT and snap.nn_ready) else "WARMUP"
        flow_label = self._flow_label(analysis.flow_vectors)
        explain = self._safe_short(analysis.explanation, 90)

        cv2.putText(frame, "Present Situation Crowd Detection", (32, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.84, (0, 235, 255), 2)
        cv2.putText(frame, f"Current: {snap.current_count}  Predicted Current: {current_nowcast}  Delta: {nowcast_delta:+d}", (32, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk: {analysis.risk_status}  Congestion: {analysis.congestion_level:.2f}  Density: {analysis.density_ratio:.3f}", (32, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (120, 255, 160), 1)
        cv2.putText(frame, f"Mode: LIVE_DETECTION  State: {state_label}  Flow: {flow_label}", (32, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1)
        cv2.putText(frame, f"Tracked IDs: {len(analysis.object_centers)}  Detected Boxes: {len(self.last_boxes)}", (32, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 225, 225), 1)
        cv2.putText(frame, f"Recommendation: {analysis.recommendation}", (32, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
        cv2.putText(frame, explain, (32, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

        if analysis.elapsed_percent < PREDICTION_START_PERCENT:
            pending = PREDICTION_START_PERCENT - analysis.elapsed_percent
            current_nowcast = self._predict_current_nowcast()
            cv2.putText(frame, "Future Crowd Prediction", (840, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 235, 255), 2)
            cv2.putText(frame, "Status: Waiting for baseline analysis", (840, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, f"Prediction unlocks at {PREDICTION_START_PERCENT:.0f}% timeline", (840, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (225, 225, 225), 1)
            cv2.putText(frame, f"Current progress: {analysis.elapsed_percent:.1f}%  |  Remaining: {pending:.1f}%", (840, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (225, 225, 225), 1)
            cv2.putText(frame, f"Current Crowd Prediction: {current_nowcast}", (840, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (220, 220, 220), 1)
            cv2.putText(frame, "Timeline prediction is enabled only after analysis threshold", (840, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)
        else:
            main_surge_label, main_surge_reason = self._surge_decision(
                snap.gate_reason,
                snap.incoming_probability_percent,
                bool(snap.incoming),
            )
            horizon_text = self._format_future_horizon(analysis.frame_idx, min(self.total_frames - 1, analysis.frame_idx + self._future_horizon_frames()))
            future_risk_text = analyze_risk(int(snap.current_count), int(snap.future_count), float(analysis.congestion_level))
            conf_band, conf_color = self._confidence_band(snap.confidence_percent)
            cv2.putText(frame, "Future Crowd Prediction", (840, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (0, 235, 255), 2)
            cv2.putText(frame, f"Now: {snap.current_count}   Future: {snap.future_count}   Delta: {snap.delta:+d}", (840, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {snap.confidence_percent:.1f}% ({conf_band})   Incoming Prob: {snap.incoming_probability_percent:.1f}%", (840, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.56, conf_color, 1)
            cv2.line(frame, (840, 126), (DISPLAY_WIDTH - 24, 126), (85, 85, 85), 1)
            cv2.putText(frame, f"Current Crowd Tracking -> Observed: {snap.current_count} | Predicted Current: {current_nowcast}", (840, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1)
            cv2.putText(frame, f"Model State: {snap.prediction_mode} | Gate: {self._safe_short(snap.gate_reason, 22)}", (840, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)
            cv2.putText(frame, f"Action: {snap.action_recommendation} | Streak: {snap.incoming_streak} | Surge: {main_surge_label}", (840, 182), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130, 255, 170), 1)
            cv2.putText(frame, f"Future Risk Status: {future_risk_text} | Horizon: {horizon_text}", (840, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
            if self.studio_forecast is not None:
                sf = self.studio_forecast
                studio_surge_label, _ = self._surge_decision(
                    sf.gate_reason,
                    sf.incoming_probability_percent,
                    sf.incoming,
                )
                studio_text = (
                    f"Studio @{sf.target_percent}%: {sf.predicted_count} ({sf.delta:+d}) risk {sf.future_risk_status} | surge {studio_surge_label}"
                )
                cv2.putText(frame, self._safe_short(studio_text, 62), (840, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 240, 170), 1)
            else:
                cv2.putText(frame, self._safe_short(f"Elapsed: {analysis.elapsed_percent:.1f}% | {main_surge_reason}", 62), (840, 222), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)

    def render(self, force_predict: bool = False) -> None:
        analysis = self.analyze_to(self.current_frame_idx)
        frame = self._get_frame_for_render(self.current_frame_idx)

        if force_predict:
            self.last_predict_message = (
                f"Predict @ frame {analysis.frame_idx}: now {analysis.count} -> future {analysis.future_count} "
                f"(confidence {analysis.snapshot.confidence_percent:.1f}%, nowcast {self._predict_current_nowcast()}, horizon {self._format_future_horizon(analysis.frame_idx, min(self.total_frames - 1, analysis.frame_idx + self._future_horizon_frames()))}) | {analysis.recommendation}"
            )

        if self.overlay_enabled:
            draw_boxes(frame, self.last_boxes)
            frame = draw_heatmap(frame, self.last_boxes)
            draw_ids(frame, analysis.object_centers)
            frame = draw_flow(frame, self.tracker)
            frame = draw_high_crowd_popup(frame, self.last_boxes, analysis.risk_status in ("HIGH", "CRITICAL", "SURGE"))

        self._overlay_custom_panels(frame, analysis)
        self._update_unique_gallery(frame)
        status_overlay = frame.copy()
        cv2.rectangle(status_overlay, (12, DISPLAY_HEIGHT - 46), (DISPLAY_WIDTH - 12, DISPLAY_HEIGHT - 8), (12, 16, 24), -1)
        cv2.addWeighted(status_overlay, 0.46, frame, 0.54, 0, frame)
        cv2.rectangle(frame, (12, DISPLAY_HEIGHT - 46), (DISPLAY_WIDTH - 12, DISPLAY_HEIGHT - 8), (0, 220, 255), 1)
        cv2.putText(frame, self._safe_short(self.last_predict_message, 132), (20, DISPLAY_HEIGHT - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)

        canvas = np.zeros((DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        canvas[:DISPLAY_HEIGHT, :] = frame
        self._draw_buttons(canvas)
        cv2.imshow(WINDOW_NAME, canvas)
        cv2.imshow(PREVIEW_WINDOW_NAME, self._build_preview_frame())

    def on_seek(self, pos: int) -> None:
        _ = pos

    def _handle_button_action(self, key: str) -> None:
        if key == "run":
            if not self.running and self.playback_segment_start_frame is None:
                self.playback_segment_start_frame = int(self.current_frame_idx)
            self.running = True
            self.last_predict_message = "Video running. Click Stop to pause, or Predict Timeline for custom forecast."
            self._log_user_operation("RUN", "Playback started/resumed")
        elif key == "stop":
            if self.running:
                self._record_pause_event(reason="Stop Button")
            self.running = False
            self.last_predict_message = "Video paused. Use Predict Timeline for explainable future simulation, or Run to continue."
            self._log_user_operation("STOP", "Playback paused by user")
        elif key == "predict":
            if self.running:
                self._record_pause_event(reason="Predict Button")
            self.running = False
            analysis = self.cache.get(self.current_frame_idx, self.analyze_to(self.current_frame_idx))
            if analysis.elapsed_percent < PREDICTION_START_PERCENT:
                self.last_predict_message = (
                    f"Prediction blocked until {PREDICTION_START_PERCENT:.0f}% analysis. "
                    f"Current: {analysis.elapsed_percent:.1f}%"
                )
                self._log_user_operation("PREDICT_BLOCKED", self.last_predict_message)
                self.render(force_predict=False)
                return
            self._log_user_operation("OPEN_PREDICT_TIMELINE", f"frame={analysis.frame_idx}, elapsed={analysis.elapsed_percent:.1f}%")
            self._open_prediction_timeline_window(analysis)
            self.render(force_predict=False)
        elif key == "overlay":
            self.overlay_enabled = not self.overlay_enabled
            self._log_user_operation("TOGGLE_OVERLAY", f"overlay={'ON' if self.overlay_enabled else 'OFF'}")
            self.render(force_predict=False)
        elif key == "reset":
            if self.running:
                self._record_pause_event(reason="Reset Button")
            self.running = False
            self.current_frame_idx = 0
            self._reset_analysis_state()
            self.studio_forecast = None
            self.studio_candidate_forecast = None
            self._log_user_operation("RESET", "Timeline rebooted to start")
            self.render(force_predict=False)
        elif key == "terminate":
            if self.running:
                self._record_pause_event(reason="Terminate Button")
            self.running = False
            self.terminate_requested = True
            self.last_predict_message = "Prediction session terminated by user."
            self._log_user_operation("TERMINATE", "Session terminated from UI button")

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.contains(x, y):
                self._handle_button_action(button.key)
                break

    def sync_trackbar(self) -> None:
        return

    def _close_and_write_summary(self) -> None:
        self.prediction_log.close()
        self.prediction_jsonl.close()

        runtime_sec = max(1.0, time.time() - self.session_started_at)
        peak_count = max(self.count_history) if self.count_history else 0
        avg_count = int(round(sum(self.count_history) / len(self.count_history))) if self.count_history else 0

        with open(PREDICTION_EXPLAIN_PATH, "w", encoding="utf-8") as f:
            f.write("# Main2 Crowd Prediction Auto-Explain\n\n")
            f.write(f"- Video: {self.video_path}\n")
            f.write(f"- Frames analyzed: {len(self.count_history)}\n")
            f.write(f"- Runtime seconds: {runtime_sec:.1f}\n")
            f.write(f"- Peak crowd: {peak_count}\n")
            f.write(f"- Average crowd: {avg_count}\n")
            f.write(f"- High-risk events: {len(self.high_risk_events)}\n")
            f.write(f"- Incoming/forecast events: {len(self.prediction_events)}\n\n")

            f.write("## Recommendations Logic\n\n")
            f.write("- IMMEDIATE_INTERVENTION: CRITICAL risk or strong model intervention signal.\n")
            f.write("- PREPARE_MARSHAL_TEAM: rising trend, surge, or high risk.\n")
            f.write("- WATCH_ENTRY_POINTS: early warning before full alert.\n")
            f.write("- NORMAL_MONITORING: stable pattern.\n\n")

            if self.high_risk_events:
                f.write("## Top High-Risk Moments\n\n")
                top_events = self.high_risk_events[-8:]
                for event in top_events:
                    snap = event.snapshot
                    f.write(
                        f"- Frame {event.frame_idx} ({event.elapsed_percent:.2f}%): current {event.count}, "
                        f"future {event.future_count}, risk {event.risk_status}, rec {event.recommendation}, "
                        f"confidence {snap.confidence_percent:.1f}%\n"
                    )
            else:
                f.write("## Top High-Risk Moments\n\n- No high-risk segments detected.\n")

        os.makedirs(PREDICTION_RESULT_TXT_DIR, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        txt_path = os.path.join(PREDICTION_RESULT_TXT_DIR, f"main2_result_{video_name}_{timestamp}.txt")
        txt_lines = [
            "Main2 Crowd Prediction Result",
            "=" * 40,
            f"Video: {self.video_path}",
            f"Frames analyzed: {len(self.count_history)}",
            f"Runtime (sec): {runtime_sec:.2f}",
            f"Peak crowd: {peak_count}",
            f"Average crowd: {avg_count}",
            f"High-risk events: {len(self.high_risk_events)}",
            f"Incoming/forecast events: {len(self.prediction_events)}",
        ]
        if self.future_count_history:
            peak_future = max(self.future_count_history)
            avg_future = int(round(sum(self.future_count_history) / len(self.future_count_history)))
            txt_lines.extend(
                [
                    f"Peak future forecast: {peak_future}",
                    f"Average future forecast: {avg_future}",
                ]
            )
        if self.future_risk_history:
            risk_counts: Dict[str, int] = {}
            for status in self.future_risk_history:
                risk_counts[status] = risk_counts.get(status, 0) + 1
            risk_line = ", ".join([f"{k}:{v}" for k, v in sorted(risk_counts.items())])
            txt_lines.append(f"Future risk distribution: {risk_line}")
        if self.count_history:
            txt_lines.extend(
                [
                    "",
                    "Recent Counts (last 20):",
                    ", ".join(str(v) for v in self.count_history[-20:]),
                ]
            )
        if self.prediction_events:
            txt_lines.extend(
                [
                    "",
                    "Recent Prediction Events (up to 10):",
                ]
            )
            for event in self.prediction_events[-10:]:
                txt_lines.append(
                    f"- Frame {event.frame_idx}, elapsed {event.elapsed_percent:.2f}%: now {event.count} -> future {event.future_count}, risk {event.risk_status}, rec {event.recommendation}"
                )

        txt_lines.extend(["", *self._build_operation_table_lines()])
        txt_lines.extend(["", *self._build_pause_result_table_lines()])

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + "\n")

        with open(PREDICTION_RESULT_TXT_LATEST, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + "\n")

        future_summary_path = os.path.join(PREDICTION_RESULT_TXT_DIR, "future_prediction_summary_main2.json")
        future_summary = {
            "video": self.video_path,
            "frames_analyzed": len(self.count_history),
            "future_forecast_series_tail": self.future_count_history[-120:],
            "future_risk_series_tail": self.future_risk_history[-120:],
            "studio_forecast": {
                "target_percent": self.studio_forecast.target_percent,
                "target_frame": self.studio_forecast.target_frame,
                "predicted_count": self.studio_forecast.predicted_count,
                "delta": self.studio_forecast.delta,
                "confidence_percent": self.studio_forecast.confidence_percent,
                "future_risk_status": self.studio_forecast.future_risk_status,
            } if self.studio_forecast is not None else None,
        }
        with open(future_summary_path, "w", encoding="utf-8") as f:
            json.dump(future_summary, f, indent=2)

        self._write_cps_upload_report(runtime_sec=runtime_sec, peak_count=peak_count, avg_count=avg_count)

        self.latest_result_txt_path = txt_path

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT)
        cv2.resizeWindow(PREVIEW_WINDOW_NAME, PREVIEW_SIZE[0], PREVIEW_SIZE[1])
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        self.render(force_predict=False)

        try:
            while True:
                loop_started = time.perf_counter()
                if self.running:
                    self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + FAST_STEP_SIZE)
                    self.render(force_predict=True)

                    render_elapsed = time.perf_counter() - loop_started
                    if self.frame_budget_sec > 0 and render_elapsed > self.frame_budget_sec * 1.8:
                        extra_steps = int(render_elapsed / self.frame_budget_sec) - 1
                        if extra_steps > 0:
                            self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + min(5, extra_steps))

                    if self.current_frame_idx >= self.total_frames - 1:
                        self._record_pause_event(reason="Video Ended")
                        self.running = False
                        self.last_predict_message = "Video ended. Click Reboot Timeline to run again from start."
                        self._log_user_operation("VIDEO_ENDED", "Reached end of video")

                if self.running:
                    spent_ms = int((time.perf_counter() - loop_started) * 1000.0)
                    target_ms = max(1, int(self.frame_budget_sec * 1000.0))
                    wait_ms = max(1, target_ms - spent_ms)
                else:
                    wait_ms = 22

                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    self._log_user_operation("QUIT", "User exited session")
                    break
                if key == ord("r"):
                    self._handle_button_action("run")
                if key == ord("s"):
                    self._handle_button_action("stop")
                if key == ord("p"):
                    self._handle_button_action("predict")
                if key == ord("o"):
                    self._handle_button_action("overlay")
                if key == ord("c"):
                    self._handle_button_action("reset")
                if key == ord("x"):
                    self._handle_button_action("terminate")
                if key == 81:
                    if self.running:
                        self._record_pause_event(reason="Seek Left")
                    self.running = False
                    self.current_frame_idx = max(0, self.current_frame_idx - 10)
                    self._log_user_operation("SEEK_LEFT", f"frame={self.current_frame_idx}")
                    self.render(force_predict=False)
                if key == 83:
                    if self.running:
                        self._record_pause_event(reason="Seek Right")
                    self.running = False
                    self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 10)
                    self._log_user_operation("SEEK_RIGHT", f"frame={self.current_frame_idx}")
                    self.render(force_predict=False)
                if self.terminate_requested:
                    break
        finally:
            if self.running:
                self._record_pause_event(reason="Session End")
            self._close_and_write_summary()
            print(f"[main2] Result TXT saved at: {self.latest_result_txt_path}")
            self.cap.release()
            cv2.destroyAllWindows()


def pick_video_file() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select Crowd Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or None
    except Exception:
        return None


def resolve_video_path(cli_video: Optional[str]) -> str:
    if cli_video and os.path.exists(cli_video):
        return cli_video

    picked = pick_video_file()
    if picked and os.path.exists(picked):
        return picked

    default_video = os.path.join("data", "crowd_videos", "crowd3.mp4")
    if os.path.exists(default_video):
        return default_video

    raise RuntimeError(
        "No video selected. Pass --video <path> or place a default file at data/crowd_videos/crowd3.mp4"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Timeline-based crowd analysis with full intelligence and advanced controls")
    parser.add_argument("--video", type=str, default=None, help="Optional path to input video")
    args = parser.parse_args()

    video_path = resolve_video_path(args.video)
    analyzer = TimelineCrowdAnalyzer(video_path)
    analyzer.run()


if __name__ == "__main__":
    main()
