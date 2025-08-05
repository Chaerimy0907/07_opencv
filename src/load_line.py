import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 크기 변환 ( 성능 최적화)
img = cv2.imread('../img/load_line.jpg')
h, w = img.shape[:2]
max_width = 600
if w > max_width:
    ratio = max_width / w
    new_h = int(h * ratio)
    img = cv2.resize(img, (max_width, new_h))

data = img.reshape((-1, 3)).astype(np.float32)

# 군집합 개수
K = 3

# KMeans 클러스터링
# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 중심값을 정수형으로 변환
center = np.uint8(center)
res = center[label.flatten()]
res = res.reshape((img.shape))

# 대표 색상 분석
labels = label.flatten()
counts = np.bincount(labels)
total_pixels = counts.sum()
ratios = counts / total_pixels