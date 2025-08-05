import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 크기 변환 ( 성능 최적화)
def resieze_image(img, max_width=600):
    h, w = img.shape[:2]
    if w > max_width:
        ratio = max_width / w
        new_h = int(h * ratio)
        img = cv2.resize(img, (max_width, new_h))
    return img
