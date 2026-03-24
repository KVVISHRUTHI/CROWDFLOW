import os
import importlib
from typing import Dict, List, Optional, Tuple

from processing.tracker import SimpleTracker

DeepSort = None
try:
    deep_sort_module = importlib.import_module("deep_sort_realtime.deepsort_tracker")
    DeepSort = getattr(deep_sort_module, "DeepSort", None)
except Exception:
    DeepSort = None


def _iou_xyxy(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class HybridTracker:
    """Hybrid tracker that keeps SimpleTracker behavior and optionally refines IDs using Deep SORT.

    This design preserves current pipeline output when Deep SORT is not available.
    """

    def __init__(self):
        self.simple_tracker = SimpleTracker()
        self.frame_index = 0
        self.crowd_mode = "low"

        self.deep_enabled = os.getenv("DEEPSORT_ENABLED", "1").strip() not in {"0", "false", "False"}
        self.deep_frame_skip = max(1, int(os.getenv("DEEPSORT_FRAME_SKIP", "1")))
        self.match_iou_threshold = float(os.getenv("DEEPSORT_SIMPLE_MATCH_IOU", "0.35"))

        self.deep_tracker = None
        self.deep_available = False
        if self.deep_enabled and DeepSort is not None:
            # Fast config: moderate age, quick confirmation, lightweight embedder.
            self.deep_tracker = DeepSort(
                max_age=20,
                n_init=2,
                max_iou_distance=0.7,
                embedder="mobilenet",
                half=True,
                bgr=True,
                embedder_gpu=False,
            )
            self.deep_available = True

        self.deep_id_to_display_id: Dict[int, int] = {}

    def set_crowd_mode(self, mode: str):
        normalized = str(mode).strip().lower()
        if normalized not in {"low", "high"}:
            return
        self.crowd_mode = normalized
        if hasattr(self.simple_tracker, "set_crowd_mode"):
            self.simple_tracker.set_crowd_mode(normalized)

    def _run_deepsort(self, detections: List[List[int]], frame) -> List[Tuple[Tuple[int, int, int, int], int]]:
        if not self.deep_available or self.deep_tracker is None:
            return []

        deep_dets = []
        for x1, y1, x2, y2 in detections:
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            deep_dets.append(([x1, y1, w, h], 1.0, "person"))

        tracks = self.deep_tracker.update_tracks(deep_dets, frame=frame)
        out = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            out.append(((x1, y1, x2, y2), int(track.track_id)))
        return out

    def _fuse_ids(
        self,
        simple_objects: List[List[int]],
        deep_tracks: List[Tuple[Tuple[int, int, int, int], int]],
    ) -> List[List[int]]:
        if not deep_tracks:
            return simple_objects

        fused: List[List[int]] = []

        for x1, y1, x2, y2, display_id in simple_objects:
            bbox = (x1, y1, x2, y2)
            best_deep_id: Optional[int] = None
            best_iou = 0.0

            for deep_bbox, deep_id in deep_tracks:
                iou = _iou_xyxy(bbox, deep_bbox)
                if iou >= self.match_iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_deep_id = deep_id

            if best_deep_id is None:
                fused.append([x1, y1, x2, y2, display_id])
                continue

            mapped_display_id = self.deep_id_to_display_id.get(best_deep_id)
            if mapped_display_id is None:
                self.deep_id_to_display_id[best_deep_id] = display_id
                mapped_display_id = display_id

            fused.append([x1, y1, x2, y2, mapped_display_id])

        return fused

    def update(self, objects_rect: List[List[int]], frame=None) -> List[List[int]]:
        self.frame_index += 1

        # Keep existing pipeline as primary tracker.
        simple_objects = self.simple_tracker.update(objects_rect, frame=frame)

        # Optional Deep SORT refinement layer.
        if (
            self.crowd_mode == "high"
            or
            not self.deep_available
            or frame is None
            or (self.frame_index % self.deep_frame_skip != 0)
        ):
            return simple_objects

        deep_tracks = self._run_deepsort(objects_rect, frame)
        return self._fuse_ids(simple_objects, deep_tracks)
