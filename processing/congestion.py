import numpy as np

def detect_congestion(boxes, frame_shape):
    h, w, _ = frame_shape

    density_map = np.zeros((h, w))

    for (x1, y1, x2, y2) in boxes:
        density_map[y1:y2, x1:x2] += 1

    max_density = np.max(density_map)

    return max_density