'''
1. 라이브러리 설치 및 임포트
2. 웹캠 연결
3. 옷 색상 데이터셋 구축
4. KNN 모델 구현
- 데이터 전처리
- KNN 모델 학습(최적 K값 탐색)
5. 실시간 옷 색상 인식 구현
- ROI 설정
- 실시간 색상 추출 및 예측
- 결과 시각화
6. 사용자 인터페이스 구현
'''

import cv2
import csv
import numpy as np

# CSV에서 학습 데이터 불러오기
samples = []
labels = []
with open('color_dataset.csv', 'r') as f:
    next(f)
    for line in f:
        r, g, b, label = line.strip().split(',')
        samples.append([int(r)/255.0, int(g)/255.0, int(b)/255.0])
        labels.append(label)

# 고유 라벨 목록 및 숫자 라벨 매핑
unique_labels = sorted(list(set(labels)))
label_to_num = {label : idx for idx, label in enumerate(unique_labels)}
num_to_label = {idx: label for label, idx in label_to_num.items()}

# 숫자 라벨로 변환
train_data = np.array(samples, dtype=np.float32)
train_labels = np.array([label_to_num[l] for l in labels], dtype=np.int32)

# KNN 모델 학습
knn = cv2.ml.KNearest_create()
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# 마우스 콜백 함수
roi = None
def mouse_callback(event, x, y, flags, param):
    global roi
    if event == cv2.EVENT_LBUTTONDOWN:
        h, w = 100, 100
        x1, y1 = max(0, x-w//2), max(0, y-h//2)
        x2, y2 = x1+w, y1+h
        roi = (x1, y1, x2, y2)

# 웹캠 설정
cap = cv2.VideoCapture(0)
cv2.namedWindow('Predict Color')
cv2.setMouseCallback('Predict Color', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if roi:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1, x2, y2), (255, 255, 255), 2)

        # ROI에서 평균 색 추출
        roi_img = frame[y1:y2, x1:x2]
        mean_color = roi_img.mean(axis=(0, 1))  # BGR
        sample = np.array([[mean_color[2]/255.0, mean_color[1]/255.0, mean_color[0]/255.0]], dtype=np.float32)

        # 예측
        ret, result, neighbors, dist = knn.findNearest(sample, k=1)
        pred_label = num_to_label[int(result[0][0])]

        # 화면에 출력
        cv2.putText(frame, f"Predicted : {pred_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Predict Color", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()