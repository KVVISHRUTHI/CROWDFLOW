import cv2
import time
import numpy as np
from collections import Counter

from models.yolo_model import load_model
from processing.video_processing import load_video, read_frame
from processing.detection import detect_people

from processing.flow_analysis import calculate_flow
from processing.congestion import detect_congestion
from processing.time_series import predict_future_crowd
from processing.risk_analysis import analyze_risk

from dashboard.visualization import (
    draw_boxes,
    draw_text,
    draw_heatmap,
    draw_ids,
    draw_flow,
    draw_global_flow_indicator,
    draw_high_crowd_popup,
)
from processing.tracker import SimpleTracker

VIDEO_PATH = "data/crowd_videos/crowd2.mp4"

DISPLAY_SIZE = (1366, 768)
DETECTION_INTERVAL = 2
ZOOM_PASS_INTERVAL = 2
PREVIEW_WINDOW_NAME = "Detected Person Preview"
PREVIEW_SIZE = (360, 300)

count_history = []


def _update_unique_gallery(frame, tracked_objects, unique_gallery):
    new_ids = []
    for x1, y1, x2, y2, person_id in tracked_objects:
        x1 = max(0, min(frame.shape[1] - 1, x1))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        x2 = max(0, min(frame.shape[1], x2))
        y2 = max(0, min(frame.shape[0], y2))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        area = (x2 - x1) * (y2 - y1)
        existing = unique_gallery.get(person_id)

        # Keep one photo per ID. Replace only if new crop is clearer/larger.
        if existing is None:
            new_ids.append(person_id)
            unique_gallery[person_id] = {
                "image": crop.copy(),
                "area": area,
            }
        elif area > existing["area"]:
            unique_gallery[person_id] = {
                "image": crop.copy(),
                "area": area,
            }

    return new_ids


def _build_preview_frame(frame, unique_gallery, ordered_ids, page_index):
    preview_w, preview_h = PREVIEW_SIZE
    canvas = 255 * (frame[:1, :1].copy())
    canvas = cv2.resize(canvas, (preview_w, preview_h))
    canvas[:, :] = (20, 20, 20)

    cv2.rectangle(canvas, (0, 0), (preview_w - 1, preview_h - 1), (0, 255, 255), 2)
    cv2.rectangle(canvas, (8, 8), (preview_w - 8, 36), (35, 35, 35), -1)
    cv2.rectangle(canvas, (8, 8), (preview_w - 8, 36), (0, 255, 255), 1)
    cv2.putText(canvas, f"Unique Persons: {len(unique_gallery)}", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    if not ordered_ids:
        cv2.putText(canvas, "No unique person yet", (70, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return canvas

    cards_per_page = 4
    total_pages = (len(ordered_ids) + cards_per_page - 1) // cards_per_page
    current_page = min(page_index, max(0, total_pages - 1))

    start = current_page * cards_per_page
    page_ids = ordered_ids[start:start + cards_per_page]

    grid_top = 48
    card_w = 160
    card_h = 118
    x_positions = [12, 188]
    y_positions = [grid_top, grid_top + card_h + 10]

    for idx, person_id in enumerate(page_ids):
        row = idx // 2
        col = idx % 2
        x = x_positions[col]
        y = y_positions[row]

        card = np.full((card_h, card_w, 3), 25, dtype=np.uint8)
        image = unique_gallery[person_id]["image"]

        available_w = card_w - 12
        available_h = card_h - 32
        scale = min(available_w / image.shape[1], available_h / image.shape[0])
        scaled = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))

        oy = 6 + (available_h - scaled.shape[0]) // 2
        ox = 6 + (available_w - scaled.shape[1]) // 2
        card[oy:oy + scaled.shape[0], ox:ox + scaled.shape[1]] = scaled

        cv2.rectangle(card, (0, 0), (card_w - 1, card_h - 1), (0, 255, 255), 1)
        cv2.rectangle(card, (0, card_h - 24), (card_w - 1, card_h - 1), (40, 40, 40), -1)
        cv2.putText(card, f"ID: {person_id}", (8, card_h - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        canvas[y:y + card_h, x:x + card_w] = card

    cv2.putText(canvas, f"Page {current_page + 1}/{total_pages}", (240, preview_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return canvas


def main():

    cv2.setUseOptimized(True)
    cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, PREVIEW_SIZE[0], PREVIEW_SIZE[1])

    model = load_model()
    cap = load_video(VIDEO_PATH)

    tracker = SimpleTracker()

    all_counts = []
    all_status = []

    frame_count = 0
    prev_time = 0
    detection_step = 0
    last_boxes = []
    last_objects = []
    unique_gallery = {}
    preview_page = 0
    preview_sequence = []

    while cap.isOpened():
        ret, frame = read_frame(cap)
        if not ret:
            break

        frame_count += 1

        frame = cv2.resize(frame, DISPLAY_SIZE)

        # 🔍 Detection
        if frame_count % DETECTION_INTERVAL == 0:
            detection_step += 1
            use_zoom = (detection_step % ZOOM_PASS_INTERVAL == 0)
            boxes = detect_people(model, frame, use_zoom=use_zoom)
            objects = tracker.update(boxes)
            last_boxes = boxes
            last_objects = objects
        else:
            boxes = last_boxes
            objects = last_objects

        # 🔁 Tracking output
        object_centers = {obj_id: ((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, obj_id in objects}
        display_centers = object_centers

        # 📊 Count + density
        tracked_count = len(objects)
        count_history.append(tracked_count)
        recent_window = count_history[-10:]
        count = int(round(sum(recent_window) / len(recent_window)))

        # 🔮 Prediction (REAL regression-based)
        future_count = predict_future_crowd(count_history)

        # 🚨 Congestion
        congestion_level = detect_congestion(boxes, frame.shape)

        # 🧠 Risk
        risk_status = analyze_risk(count, future_count, congestion_level)

        all_counts.append(count)
        all_status.append(risk_status)

        # 🎨 DRAWING SECTION

        draw_boxes(frame, boxes)
        frame = draw_heatmap(frame, boxes)
        draw_ids(frame, display_centers)
        frame = draw_flow(frame, tracker)

        draw_text(frame, count, future_count, risk_status)

        # 🔥 TITLE
        cv2.putText(frame, "AI CROWD INTELLIGENCE SYSTEM",
                    (120, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        # 🔥 BORDER
        cv2.rectangle(frame, (0, 0),
                      (frame.shape[1], frame.shape[0]),
                      (0, 255, 255), 2)

        # 🔥 FLOW DIRECTION ANALYSIS
        flow = calculate_flow(tracker)
        frame = draw_global_flow_indicator(frame, flow)

        # Dynamic hotspot popup appears at the densest crowd area in the frame.
        show_hotspot_popup = risk_status in ("HIGH", "CRITICAL")
        frame = draw_high_crowd_popup(frame, boxes, show_hotspot_popup)

        # ⚡ FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {int(fps)}",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)

        cv2.imshow("Crowd Intelligence System", frame)

        new_ids = _update_unique_gallery(frame, objects, unique_gallery)
        if new_ids:
            preview_sequence.extend(new_ids)

        if frame_count % 20 == 0 and preview_sequence:
            max_page = max(0, (len(preview_sequence) - 1) // 4)
            preview_page = min(preview_page + 1, max_page)

        preview_frame = _build_preview_frame(frame, unique_gallery, preview_sequence, preview_page)
        cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 📊 FINAL ANALYSIS
    if len(all_counts) > 0:
        max_count = max(all_counts)
        avg_count = sum(all_counts) // len(all_counts)
        final_status = Counter(all_status).most_common(1)[0][0]

        print("\n===== FINAL VIDEO ANALYSIS =====")
        print(f"Max Crowd Count: {max_count}")
        print(f"Average Crowd Count: {avg_count}")
        print(f"Final Risk Status: {final_status}")
    else:
        print("\n⚠ No frames processed!")

    # 📄 SAVE UNIQUE PEOPLE
    with open("crowd_summary.txt", "w") as f:
        summary_rows = [
            (display_id, cx, cy)
            for display_id, (cx, cy) in tracker.display_first_seen_positions.items()
        ]

        for display_id, cx, cy in sorted(summary_rows, key=lambda row: row[0]):
            f.write(f"ID {display_id} first seen at ({cx},{cy})\n")

    print(f"\nTotal Unique People Detected: {len(tracker.display_first_seen_positions)}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()