import math
from typing import Dict, Optional, Tuple

import numpy as np

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.objects = {}
        self.previous_positions = {}
        self.id_count = 1
        self.unique_ids = set()

        self.display_id_map = {}
        self.next_display_id = 1
        self.display_first_seen_positions = {}
        self.retired_tracks = []
        self.identity_memory = {}
        self.frame_index = 0

        self.max_distance = 75
        self.max_missed = 18
        self.min_hits = 3
        self.min_unique_hits = 6
        self.min_unique_area = 500
        self.layer_height = 80

        # Re-identification settings to reduce duplicate unique IDs.
        self.reid_max_age = 90
        self.reid_max_distance = 120
        self.reid_min_area_ratio = 0.45
        self.reid_appearance_weight = 45.0
        self.reid_max_signature_distance = 0.22

        # Long-gap memory to handle short-loop videos where persons reappear after full rotation.
        self.memory_max_age = 1800
        self.memory_max_center_distance = 180
        self.memory_min_area_ratio = 0.40
        self.memory_max_signature_distance = 0.18

        self.crowd_mode = "low"
        self.enable_appearance_reid = True
        self.enable_long_memory_reid = True

    def set_crowd_mode(self, mode: str):
        normalized = str(mode).strip().lower()
        if normalized not in {"low", "high"}:
            return

        self.crowd_mode = normalized
        if normalized == "high":
            # High-crowd fallback: keep legacy stable behavior.
            self.enable_appearance_reid = False
            self.enable_long_memory_reid = False
        else:
            self.enable_appearance_reid = True
            self.enable_long_memory_reid = True

    @staticmethod
    def _bbox_area(x1, y1, x2, y2):
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _prune_retired_tracks(self):
        self.retired_tracks = [
            item for item in self.retired_tracks
            if self.frame_index - item["retired_at"] <= self.reid_max_age
        ]

    def _prune_identity_memory(self):
        if not self.enable_long_memory_reid:
            return
        keys_to_drop = []
        for display_id, info in self.identity_memory.items():
            if self.frame_index - info["last_seen_at"] > self.memory_max_age:
                keys_to_drop.append(display_id)
        for display_id in keys_to_drop:
            del self.identity_memory[display_id]

    @staticmethod
    def _clamp_bbox_to_frame(bbox: Tuple[int, int, int, int], frame_shape):
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _extract_signature(self, frame, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if not self.enable_appearance_reid:
            return None
        if frame is None:
            return None
        clamped = self._clamp_bbox_to_frame(bbox, frame.shape)
        if clamped is None:
            return None
        x1, y1, x2, y2 = clamped
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Non-sensitive appearance descriptor (color + texture spread), no biometrics.
        mean = crop.mean(axis=(0, 1)).astype(np.float32) / 255.0
        std = crop.std(axis=(0, 1)).astype(np.float32) / 255.0
        sig = np.concatenate([mean, std], axis=0)
        norm = float(np.linalg.norm(sig))
        if norm <= 1e-8:
            return None
        return sig / norm

    @staticmethod
    def _signature_distance(sig_a: Optional[np.ndarray], sig_b: Optional[np.ndarray]) -> float:
        if sig_a is None or sig_b is None:
            return 1.0
        return float(np.linalg.norm(sig_a - sig_b))

    def _try_reidentify_display_id(self, detection):
        x1, y1, x2, y2, cx, cy, signature = detection
        det_area = self._bbox_area(x1, y1, x2, y2)

        if det_area == 0 or not self.retired_tracks:
            return None

        best_idx = None
        best_score = None

        for idx, item in enumerate(self.retired_tracks):
            px, py = item["center"]
            dist = math.hypot(cx - px, cy - py)
            if dist > self.reid_max_distance:
                continue

            prev_area = item["area"]
            if prev_area == 0:
                continue

            area_ratio = min(det_area, prev_area) / max(det_area, prev_area)
            if area_ratio < self.reid_min_area_ratio:
                continue

            sig_distance = 0.0
            if self.enable_appearance_reid:
                sig_distance = self._signature_distance(signature, item.get("signature"))
                if sig_distance > self.reid_max_signature_distance:
                    continue

            score = dist + (1.0 - area_ratio) * 35.0 + sig_distance * self.reid_appearance_weight
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return None

        matched = self.retired_tracks.pop(best_idx)
        return matched["display_id"]

    def _try_memory_reidentify_display_id(self, center, area, signature):
        if not self.enable_long_memory_reid:
            return None
        cx, cy = center
        best_display_id = None
        best_score = None

        for display_id, info in self.identity_memory.items():
            px, py = info["last_center"]
            dist = math.hypot(cx - px, cy - py)
            if dist > self.memory_max_center_distance:
                continue

            prev_area = info["last_area"]
            if prev_area <= 0 or area <= 0:
                continue
            area_ratio = min(area, prev_area) / max(area, prev_area)
            if area_ratio < self.memory_min_area_ratio:
                continue

            sig_distance = self._signature_distance(signature, info.get("signature"))
            if sig_distance > self.memory_max_signature_distance:
                continue

            score = dist + (1.0 - area_ratio) * 40.0 + sig_distance * 60.0
            if best_score is None or score < best_score:
                best_score = score
                best_display_id = display_id

        return best_display_id

    def _update_identity_memory(self, display_id, center, area, signature):
        if not self.enable_long_memory_reid:
            return
        previous = self.identity_memory.get(display_id)
        if previous is not None and signature is None:
            signature = previous.get("signature")
        self.identity_memory[display_id] = {
            "last_center": center,
            "last_area": area,
            "signature": signature,
            "last_seen_at": self.frame_index,
        }

    def update(self, objects_rect, frame=None):
        self.frame_index += 1
        self._prune_retired_tracks()
        self._prune_identity_memory()

        detections = []
        for x1, y1, x2, y2 in objects_rect:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            signature = self._extract_signature(frame, (x1, y1, x2, y2))
            detections.append((x1, y1, x2, y2, cx, cy, signature))

        self.previous_positions = self.objects.copy()

        matched_track_ids = set()
        matched_detection_ids = set()

        # Greedy nearest-center matching for stable IDs.
        while True:
            best_pair = None
            best_dist = None

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue

                tx, ty = track["center"]
                for det_idx, det in enumerate(detections):
                    if det_idx in matched_detection_ids:
                        continue

                    _, _, _, _, cx, cy, _ = det
                    dist = math.hypot(cx - tx, cy - ty)
                    if dist <= self.max_distance and (best_dist is None or dist < best_dist):
                        best_dist = dist
                        best_pair = (track_id, det_idx)

            if best_pair is None:
                break

            track_id, det_idx = best_pair
            x1, y1, x2, y2, cx, cy, signature = detections[det_idx]
            track = self.tracks[track_id]
            track["bbox"] = (x1, y1, x2, y2)
            track["center"] = (cx, cy)
            track["missed"] = 0
            track["hits"] += 1
            if signature is not None:
                prev_sig = track.get("signature")
                if prev_sig is None:
                    track["signature"] = signature
                else:
                    track["signature"] = (0.7 * prev_sig) + (0.3 * signature)
                    norm = float(np.linalg.norm(track["signature"]))
                    if norm > 1e-8:
                        track["signature"] = track["signature"] / norm

            matched_track_ids.add(track_id)
            matched_detection_ids.add(det_idx)

        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_track_ids:
                track["missed"] += 1
                if track["missed"] > self.max_missed:
                    display_id = self.display_id_map.pop(track_id, None)
                    if display_id is not None:
                        x1, y1, x2, y2 = track["bbox"]
                        self.retired_tracks.append({
                            "display_id": display_id,
                            "center": track["center"],
                            "area": self._bbox_area(x1, y1, x2, y2),
                            "signature": track.get("signature"),
                            "retired_at": self.frame_index,
                        })
                    del self.tracks[track_id]

        for det_idx, det in enumerate(detections):
            if det_idx in matched_detection_ids:
                continue

            x1, y1, x2, y2, cx, cy, signature = det
            restored_display_id = self._try_reidentify_display_id(det)
            self.tracks[self.id_count] = {
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "missed": 0,
                "hits": 1,
                "signature": signature,
            }
            if restored_display_id is not None:
                self.display_id_map[self.id_count] = restored_display_id
                self.unique_ids.add(restored_display_id)
            self.id_count += 1

        confirmed_tracks = []
        for track_id, track in self.tracks.items():
            if track["missed"] == 0 and track["hits"] >= self.min_hits:
                x1, y1, x2, y2 = track["bbox"]
                cx, cy = track["center"]
                area = self._bbox_area(x1, y1, x2, y2)
                confirmed_tracks.append((track_id, x1, y1, x2, y2, cx, cy, track["hits"], area))

        new_confirmed = [
            item for item in confirmed_tracks
            if item[0] not in self.display_id_map
            and item[7] >= self.min_unique_hits
            and item[8] >= self.min_unique_area
        ]
        # Front rows first (larger y), then move to back rows.
        new_confirmed.sort(key=lambda item: (-(item[6] // self.layer_height), item[5], -item[6]))

        for track_id, x1, y1, x2, y2, cx, cy, _, _ in new_confirmed:
            area = self._bbox_area(x1, y1, x2, y2)
            signature = self.tracks[track_id].get("signature")
            reused_display_id = self._try_memory_reidentify_display_id((cx, cy), area, signature)
            if reused_display_id is not None:
                display_id = reused_display_id
            else:
                display_id = self.next_display_id
                self.next_display_id += 1

            self.display_id_map[track_id] = display_id
            self.display_first_seen_positions[display_id] = (cx, cy)
            self.unique_ids.add(display_id)

        objects_bbs_ids = []
        current_objects = {}
        for track_id, x1, y1, x2, y2, cx, cy, _, _ in confirmed_tracks:
            display_id = self.display_id_map.get(track_id)
            if display_id is None:
                continue
            objects_bbs_ids.append([x1, y1, x2, y2, display_id])
            current_objects[display_id] = (cx, cy)
            self._update_identity_memory(
                display_id,
                center=(cx, cy),
                area=self._bbox_area(x1, y1, x2, y2),
                signature=self.tracks[track_id].get("signature"),
            )

        self.objects = current_objects
        return objects_bbs_ids