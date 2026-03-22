import cv2
import numpy as np

# 🎨 COLOR SETTINGS (EDIT HERE IF NEEDED)

BOX_COLOR = (0, 255, 0)        # Green boxes
COUNT_COLOR = (0, 0, 255)      # Red text
PREDICT_COLOR = (255, 0, 0)    # Blue text
STATUS_COLOR = (0, 255, 255)   # Yellow text
FPS_COLOR = (255, 255, 0)      # Cyan text


# 🟩 DRAW BOXES
def draw_boxes(frame, boxes):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)


# 🔥 DRAW HEATMAP (SMART COLOR BASED ON CROWD)

def draw_heatmap(frame, boxes):

    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x1, y1, x2, y2) in boxes:
        heatmap[y1:y2, x1:x2] += 1

    heatmap = cv2.GaussianBlur(heatmap, (31, 31), 0)

    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 🔥 LIGHTER OVERLAY (KEY FIX)
    output = cv2.addWeighted(frame, 0.85, heatmap, 0.15, 0)

    return output

def draw_ids(frame, objects):

    for obj_id, (cx, cy) in objects.items():
        cv2.putText(frame, f"ID {obj_id}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_text(frame, count, future, status):

    # 🔥 PANEL BACKGROUND
    cv2.rectangle(frame, (10, 10), (300, 140), (20, 20, 20), -1)

    # BORDER
    cv2.rectangle(frame, (10, 10), (300, 140), (0, 255, 255), 2)

    # TEXT
    cv2.putText(frame, f"Current: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Future: {future}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 🔥 COLOR BASED STATUS
    if status == "SAFE":
        color = (0, 255, 0)
    elif status == "HIGH":
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.putText(frame, f"Status: {status}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_flow(frame, tracker):
    min_motion = 2
    arrow_scale = 4

    for obj_id in tracker.objects:
        if obj_id in tracker.previous_positions:

            x1, y1 = tracker.previous_positions[obj_id]
            x2, y2 = tracker.objects[obj_id]

            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) + abs(dy) < min_motion:
                continue

            end_x = int(x2 + dx * arrow_scale)
            end_y = int(y2 + dy * arrow_scale)

            end_x = max(0, min(frame.shape[1] - 1, end_x))
            end_y = max(0, min(frame.shape[0] - 1, end_y))

            cv2.circle(frame, (x1, y1), 3, (0, 255, 255), -1)

            cv2.arrowedLine(
                frame,
                (x1, y1),
                (end_x, end_y),
                (0, 215, 255),
                2,
                tipLength=0.35
            )

    return frame


def draw_global_flow_indicator(frame, flow_vectors):
    if not flow_vectors:
        return frame

    avg_dx = sum(v[0] for v in flow_vectors) / len(flow_vectors)
    avg_dy = sum(v[1] for v in flow_vectors) / len(flow_vectors)

    if abs(avg_dx) < 1 and abs(avg_dy) < 1:
        label = "Flow: STABLE"
        color = (200, 200, 200)
    elif abs(avg_dx) >= abs(avg_dy):
        label = "Flow: RIGHT" if avg_dx > 0 else "Flow: LEFT"
        color = (255, 255, 255)
    else:
        label = "Flow: DOWN" if avg_dy > 0 else "Flow: UP"
        color = (255, 255, 255)

    cv2.rectangle(frame, (320, 22), (640, 82), (20, 20, 20), -1)
    cv2.rectangle(frame, (320, 22), (640, 82), (0, 255, 255), 2)
    cv2.putText(frame, label, (340, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    if "RIGHT" in label or "LEFT" in label:
        start = (570, 50)
        end = (625, 50) if "RIGHT" in label else (515, 50)
    elif "DOWN" in label:
        start = (570, 35)
        end = (570, 75)
    elif "UP" in label:
        start = (570, 75)
        end = (570, 35)
    else:
        start = end = None

    if start and end:
        cv2.arrowedLine(frame, start, end, (0, 255, 255), 3, tipLength=0.35)

    return frame


def draw_high_crowd_popup(frame, boxes, show_popup):
    if not show_popup or not boxes:
        return frame

    h, w = frame.shape[:2]
    cell = 40
    grid_h = max(1, h // cell)
    grid_w = max(1, w // cell)
    grid = np.zeros((grid_h, grid_w), dtype=np.int32)

    for x1, y1, x2, y2 in boxes:
        gx1 = max(0, min(grid_w - 1, x1 // cell))
        gy1 = max(0, min(grid_h - 1, y1 // cell))
        gx2 = max(0, min(grid_w - 1, x2 // cell))
        gy2 = max(0, min(grid_h - 1, y2 // cell))
        grid[gy1:gy2 + 1, gx1:gx2 + 1] += 1

    gy, gx = np.unravel_index(np.argmax(grid), grid.shape)
    max_density = grid[gy, gx]

    if max_density < 2:
        return frame

    cx = int((gx + 0.5) * cell)
    cy = int((gy + 0.5) * cell)

    zone_w = 180
    zone_h = 130
    x1 = max(0, cx - zone_w // 2)
    y1 = max(0, cy - zone_h // 2)
    x2 = min(w - 1, x1 + zone_w)
    y2 = min(h - 1, y1 + zone_h)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    label_y = max(25, y1 - 12)
    cv2.putText(frame, "HIGH CROWD ZONE", (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return frame