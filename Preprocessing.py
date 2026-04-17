import cv2
import numpy as np


def preprocess_image(image):
    cropped = image[25:425, 25:575]
    resized = cv2.resize(cropped, (224, 224))

    normalized = resized / 255.0
    normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return normalized.astype(np.float32)