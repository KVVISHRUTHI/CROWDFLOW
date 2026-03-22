def improved_count(boxes, density):

    yolo_count = len(boxes)

    #  Convert density to approximate people count
    density_count = int(density / 500)

    #  Hybrid count
    final_count = max(yolo_count, density_count)

    return final_count