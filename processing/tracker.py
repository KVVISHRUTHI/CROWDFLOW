import math

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

    @staticmethod
    def _bbox_area(x1, y1, x2, y2):
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _prune_retired_tracks(self):
        self.retired_tracks = [
            item for item in self.retired_tracks
            if self.frame_index - item["retired_at"] <= self.reid_max_age
        ]

    def _try_reidentify_display_id(self, detection):
        x1, y1, x2, y2, cx, cy = detection
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

            score = dist + (1.0 - area_ratio) * 35.0
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return None

        matched = self.retired_tracks.pop(best_idx)
        return matched["display_id"]

    def update(self, objects_rect):
        self.frame_index += 1
        self._prune_retired_tracks()

        detections = []
        for x1, y1, x2, y2 in objects_rect:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detections.append((x1, y1, x2, y2, cx, cy))

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

                    _, _, _, _, cx, cy = det
                    dist = math.hypot(cx - tx, cy - ty)
                    if dist <= self.max_distance and (best_dist is None or dist < best_dist):
                        best_dist = dist
                        best_pair = (track_id, det_idx)

            if best_pair is None:
                break

            track_id, det_idx = best_pair
            x1, y1, x2, y2, cx, cy = detections[det_idx]
            track = self.tracks[track_id]
            track["bbox"] = (x1, y1, x2, y2)
            track["center"] = (cx, cy)
            track["missed"] = 0
            track["hits"] += 1

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
                            "retired_at": self.frame_index,
                        })
                    del self.tracks[track_id]

        for det_idx, det in enumerate(detections):
            if det_idx in matched_detection_ids:
                continue

            x1, y1, x2, y2, cx, cy = det
            restored_display_id = self._try_reidentify_display_id(det)
            self.tracks[self.id_count] = {
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "missed": 0,
                "hits": 1,
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

        for track_id, _, _, _, _, cx, cy, _, _ in new_confirmed:
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

        self.objects = current_objects
        return objects_bbs_ids