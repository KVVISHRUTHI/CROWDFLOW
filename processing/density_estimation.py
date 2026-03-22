import cv2
import numpy as np

def calculate_density(frame, boxes):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge-based density (detect crowd texture)
    edges = cv2.Canny(gray, 50, 150)

    density_score = np.sum(edges) / 255

    return density_score