# OpenCV 실습


```python
load_line.py

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
K = 8

# KMeans 클러스터링
# 반복 중지 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 중심값을 정수형으로 변환
center = np.uint8(center)

# 각 레이블에 해당하는 중심값으로 픽셀 값 선택
res = center[label.flatten()]

# 원본 영상의 형태로 변환
res = res.reshape((img.shape))

# 대표 색상 분석
labels = label.flatten()
counts = np.bincount(labels)
total_pixels = counts.sum()
ratios = counts / total_pixels

# 분석 결과 출력
print("\n[색상 분석 결과]")
for i in range(K):
    print(f"{i+1}.BGR : {center[i].tolist()} / 픽셀 수 : {counts[i]} / 비율 : {ratios[i] : .2%}")

# 색상 이미지 생성
palette = np.zeros((50, 100 * K, 3), dtype=np.uint8)
for i in range(K):
    palette[:, i*100:(i+1)*100] = center[i]

# 시각화
merged = np.hstack((img, res))  # 원본 + 색상 단순화 이미지
cv2.imshow('KMeans Result', merged)
cv2.imshow('Color Palette', palette)

# Matplotlib 파이차트
colors_rgb = [center[i][::-1] / 255.0 for i in range(K)]    # BGR -> RGB
label_text = [f'{i+1}' for i in range(K)]

plt.figure(figsize=(5,5))
plt.pie(ratios, labels= label_text, colors=colors_rgb, autopct='%1.1f%%')
plt.title('Ratios')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
```

---

  
<img width="554" height="254" alt="7_4" src="https://github.com/user-attachments/assets/08e7cf7b-2175-4a52-b98a-11c6bcbb67ad" />  

<img width="1193" height="497" alt="7_3" src="https://github.com/user-attachments/assets/04bea293-9993-4868-ae1e-fd156a02915c" />  

<img width="502" height="568" alt="7_1" src="https://github.com/user-attachments/assets/9c4e27a1-2dd2-4b48-9c85-ad24869dd93a" />
