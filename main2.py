import argparse
import json
import os
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

WINDOW_NAME = "Crowd Timeline Predictor"
BUTTON_BAR_HEIGHT = 120
DISPLAY_WIDTH = 1366
DISPLAY_HEIGHT = 768
DETECTION_INTERVAL = 2
ZOOM_PASS_INTERVAL = 2
PREDICTION_METRICS_PATH = "models/crowd_predictor_metrics.txt"
PREDICTION_CSV_PATH = "prediction_output_log_main2.csv"
PREDICTION_JSONL_PATH = "prediction_output_log_main2.jsonl"
PREDICTION_EXPLAIN_PATH = "prediction_recommendations_main2.md"
PREDICTION_START_PERCENT = 0.0
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
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
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
        self.density_history: List[float] = []
        self.action_history: List[List[float]] = []
        self.attribute_history: List[List[float]] = []
        self.incoming_streak = 0
        self.previous_tracked_centers: Dict[int, Tuple[int, int]] = {}

        self.last_boxes: List[List[int]] = []
        self.last_objects: List[List[int]] = []
        self.detection_step = 0

        self.cache: Dict[int, FrameAnalysis] = {}
        self.last_analyzed_frame = -1

        self.current_frame_idx = 0
        self.running = False
        self.overlay_enabled = True
        self.auto_predict_enabled = True
        self.last_predict_message = "Move timeline and click Predict"

        self._ignore_trackbar_callback = False
        self.buttons = self._build_buttons()

        self.session_started_at = time.time()
        self.high_risk_events: List[FrameAnalysis] = []
        self.prediction_events: List[FrameAnalysis] = []
        self.prediction_log = open(PREDICTION_CSV_PATH, "w", encoding="utf-8")
        self.prediction_jsonl = open(PREDICTION_JSONL_PATH, "w", encoding="utf-8")
        self.prediction_log.write(
            "frame,elapsed_percent,current_count,future_count,delta,confidence_percent,incoming_probability_percent,"
            "incoming,risk_status,congestion,flow_label,recommendation,gate_reason,action_recommendation,prediction_mode\n"
        )

    def _build_buttons(self) -> List[Button]:
        y1 = DISPLAY_HEIGHT + 18
        y2 = DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT - 20
        return [
            Button("run", 20, y1, 160, y2),
            Button("stop", 180, y1, 320, y2),
            Button("predict", 340, y1, 520, y2),
            Button("overlay", 540, y1, 760, y2),
            Button("auto", 780, y1, 1000, y2),
            Button("reset", 1020, y1, 1200, y2),
        ]

    def _button_label(self, key: str) -> str:
        if key == "run":
            return "Run"
        if key == "stop":
            return "Stop"
        if key == "predict":
            return "Predict"
        if key == "overlay":
            return f"Overlay: {'ON' if self.overlay_enabled else 'OFF'}"
        if key == "auto":
            return f"AutoPredict: {'ON' if self.auto_predict_enabled else 'OFF'}"
        if key == "reset":
            return "Reset Timeline"
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
        return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    def _reset_analysis_state(self) -> None:
        self.cache.clear()
        self.count_history.clear()
        self.density_history.clear()
        self.action_history.clear()
        self.attribute_history.clear()
        self.incoming_streak = 0
        self.previous_tracked_centers.clear()
        self.last_boxes = []
        self.last_objects = []
        self.last_analyzed_frame = -1
        self.detection_step = 0
        self.smooth_detect_count = 0.0
        self.tracker_mode = "low"
        try:
            self.tracker = HybridTracker()
        except Exception:
            self.tracker = SimpleTracker()
        _set_tracker_mode_compat(self.tracker, self.tracker_mode)

    def _analyze_frame(self, frame_idx: int) -> FrameAnalysis:
        frame = self._read_frame(frame_idx)

        if frame_idx % DETECTION_INTERVAL == 0:
            self.detection_step += 1
            use_zoom = (self.detection_step % ZOOM_PASS_INTERVAL == 0)
            boxes = detect_people(self.model, frame, use_zoom=use_zoom)

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

            objects = _update_tracker_compat(self.tracker, boxes, frame)
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

        frame_area = max(1, frame.shape[0] * frame.shape[1])
        box_area_sum = 0
        for x1, y1, x2, y2 in boxes:
            box_area_sum += max(0, x2 - x1) * max(0, y2 - y1)
        density_ratio = min(1.0, box_area_sum / frame_area)
        self.density_history.append(float(density_ratio))

        action_features, attribute_features, self.previous_tracked_centers = _compute_action_attribute_features(
            objects,
            boxes,
            self.previous_tracked_centers,
            frame.shape,
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
            nn_pred = self.predictor.predict(
                self.count_history,
                self.density_history,
                elapsed_ratio,
                action_features=action_features,
                attribute_features=attribute_features,
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

        congestion_level = detect_congestion(boxes, frame.shape)
        risk_status = analyze_risk(count, future_count, congestion_level)
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
            f"{frame_idx},{elapsed_percent:.3f},{snapshot.current_count},{snapshot.future_count},{snapshot.delta},"
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
        for button in self.buttons:
            active = False
            if button.key == "run" and self.running:
                active = True
            if button.key == "stop" and not self.running:
                active = True
            if button.key == "overlay" and self.overlay_enabled:
                active = True
            if button.key == "auto" and self.auto_predict_enabled:
                active = True

            fill = (0, 180, 255) if active else (40, 40, 40)
            text_col = (20, 20, 20) if active else (255, 255, 255)
            cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), fill, -1)
            cv2.rectangle(canvas, (button.x1, button.y1), (button.x2, button.y2), (255, 255, 255), 2)
            cv2.putText(
                canvas,
                self._button_label(button.key),
                (button.x1 + 12, button.y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                text_col,
                2,
            )

        cv2.putText(
            canvas,
            "Keys: r=Run s=Stop p=Predict o=Overlay a=AutoPredict c=Reset q=Quit",
            (20, DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
        )

    def _overlay_custom_panels(self, frame: np.ndarray, analysis: FrameAnalysis) -> None:
        snap = analysis.snapshot
        cv2.putText(frame, "CROWD TIMELINE ANALYZER - ADVANCED", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2)
        cv2.putText(frame, f"Recommendation: {analysis.recommendation}", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 255), 2)
        cv2.putText(frame, f"Explain: {analysis.explanation}", (20, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1)
        cv2.putText(frame, f"Risk: {analysis.risk_status}  Congestion: {analysis.congestion_level:.1f}", (20, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        frame_sec = self._seconds_from_frame(analysis.frame_idx)
        total_sec = self._seconds_from_frame(self.total_frames - 1)
        cv2.putText(
            frame,
            f"Timeline: {self._format_time(frame_sec)} / {self._format_time(total_sec)} | Frame: {analysis.frame_idx}/{self.total_frames - 1}",
            (20, DISPLAY_HEIGHT - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Predict Summary: now {snap.current_count} -> future {snap.future_count}, confidence {snap.confidence_percent:.1f}%",
            (20, DISPLAY_HEIGHT - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

    def render(self, force_predict: bool = False) -> None:
        analysis = self.analyze_to(self.current_frame_idx)
        frame = self._read_frame(self.current_frame_idx)

        if force_predict:
            self.last_predict_message = (
                f"Predict @ frame {analysis.frame_idx}: now {analysis.count} -> future {analysis.future_count} "
                f"(confidence {analysis.snapshot.confidence_percent:.1f}%) | {analysis.recommendation}"
            )

        if self.overlay_enabled:
            draw_boxes(frame, self.last_boxes)
            frame = draw_heatmap(frame, self.last_boxes)
            draw_ids(frame, analysis.object_centers)
            frame = draw_flow(frame, self.tracker)
            draw_text(frame, analysis.count, analysis.future_count, analysis.risk_status)
            frame = draw_global_flow_indicator(frame, analysis.flow_vectors)
            frame = draw_high_crowd_popup(frame, self.last_boxes, analysis.risk_status in ("HIGH", "CRITICAL", "SURGE"))
            frame = draw_future_prediction_popup(frame, analysis.snapshot, analysis.elapsed_percent, True)

        self._overlay_custom_panels(frame, analysis)
        cv2.putText(frame, self.last_predict_message, (20, 136), cv2.FONT_HERSHEY_SIMPLEX, 0.57, (0, 255, 255), 2)

        canvas = np.zeros((DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        canvas[:DISPLAY_HEIGHT, :] = frame
        self._draw_buttons(canvas)
        cv2.imshow(WINDOW_NAME, canvas)

    def on_seek(self, pos: int) -> None:
        if self._ignore_trackbar_callback:
            return
        self.current_frame_idx = max(0, min(self.total_frames - 1, int(pos)))
        self.running = False
        self.render(force_predict=False)

    def _handle_button_action(self, key: str) -> None:
        if key == "run":
            self.running = True
        elif key == "stop":
            self.running = False
        elif key == "predict":
            self.running = False
            self.render(force_predict=True)
        elif key == "overlay":
            self.overlay_enabled = not self.overlay_enabled
            self.render(force_predict=False)
        elif key == "auto":
            self.auto_predict_enabled = not self.auto_predict_enabled
            self.render(force_predict=False)
        elif key == "reset":
            self.running = False
            self.current_frame_idx = 0
            self._reset_analysis_state()
            self.sync_trackbar()
            self.render(force_predict=False)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for button in self.buttons:
            if button.contains(x, y):
                self._handle_button_action(button.key)
                break

    def sync_trackbar(self) -> None:
        self._ignore_trackbar_callback = True
        cv2.setTrackbarPos("Timeline", WINDOW_NAME, int(self.current_frame_idx))
        self._ignore_trackbar_callback = False

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

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT + BUTTON_BAR_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        cv2.createTrackbar("Timeline", WINDOW_NAME, 0, max(1, self.total_frames - 1), self.on_seek)
        self.render(force_predict=False)
        self.sync_trackbar()

        try:
            while True:
                if self.running:
                    self.current_frame_idx = min(self.total_frames - 1, self.current_frame_idx + 1)
                    self.render(force_predict=self.auto_predict_enabled)
                    self.sync_trackbar()
                    if self.current_frame_idx >= self.total_frames - 1:
                        self.running = False

                key = cv2.waitKey(8 if self.running else 30) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self._handle_button_action("run")
                if key == ord("s"):
                    self._handle_button_action("stop")
                if key == ord("p"):
                    self._handle_button_action("predict")
                if key == ord("o"):
                    self._handle_button_action("overlay")
                if key == ord("a"):
                    self._handle_button_action("auto")
                if key == ord("c"):
                    self._handle_button_action("reset")
        finally:
            self._close_and_write_summary()
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
