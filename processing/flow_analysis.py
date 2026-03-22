def calculate_flow(tracker):
    directions = []

    for obj_id in tracker.objects:
        if obj_id in tracker.previous_positions:

            x1, y1 = tracker.previous_positions[obj_id]
            x2, y2 = tracker.objects[obj_id]

            dx = x2 - x1
            dy = y2 - y1

            directions.append((dx, dy))

    return directions