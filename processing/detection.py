import cv2


def _collect_person_boxes(model, image, conf, min_area):
    results = model(image, conf=conf, iou=0.5, verbose=False)
    out = []
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) != 0:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w * h >= min_area:
                out.append([x1, y1, x2, y2])
    return out


def _nms_xyxy(boxes, iou_threshold=0.5):
    if not boxes:
        return []

    cv2_boxes = []
    scores = []
    for x1, y1, x2, y2 in boxes:
        cv2_boxes.append([x1, y1, max(1, x2 - x1), max(1, y2 - y1)])
        scores.append(1.0)

    keep = cv2.dnn.NMSBoxes(cv2_boxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
    if len(keep) == 0:
        return []

    keep_idx = [int(i[0]) if isinstance(i, (list, tuple)) else int(i) for i in keep]
    return [boxes[i] for i in keep_idx]


def detect_people(model, frame, use_zoom=True):
    h, w = frame.shape[:2]

    # Pass-1: full frame for medium/large persons.
    full_boxes = _collect_person_boxes(model, frame, conf=0.22, min_area=800)

    if not use_zoom:
        return _nms_xyxy(full_boxes, iou_threshold=0.4)

    # Pass-2: zoomed upper area to recover small/distant back-row persons.
    top_h = int(h * 0.62)
    top_crop = frame[:top_h, :]
    zoom = 1.6
    zoomed = cv2.resize(top_crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    zoom_boxes = _collect_person_boxes(model, zoomed, conf=0.16, min_area=320)

    mapped_zoom_boxes = []
    for x1, y1, x2, y2 in zoom_boxes:
        ox1 = int(x1 / zoom)
        oy1 = int(y1 / zoom)
        ox2 = int(x2 / zoom)
        oy2 = int(y2 / zoom)
        ox1 = max(0, min(w - 1, ox1))
        oy1 = max(0, min(top_h - 1, oy1))
        ox2 = max(0, min(w, ox2))
        oy2 = max(0, min(top_h, oy2))
        if ox2 > ox1 and oy2 > oy1:
            mapped_zoom_boxes.append([ox1, oy1, ox2, oy2])

    merged = full_boxes + mapped_zoom_boxes
    return _nms_xyxy(merged, iou_threshold=0.4)