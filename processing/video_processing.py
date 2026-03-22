def load_video(path):
    import cv2
    return cv2.VideoCapture(path)

def read_frame(cap):
    return cap.read()