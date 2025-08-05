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

# 웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

win_name = 'Color Prediction'

roi = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

        roi_area = frame[y:y+h, x:x+w]
        avg_color = cv2.mean(roi_area)[:3]
        r, g, b = avg_color[2], avg_color[1], avg_color[0]
        input_data = np.array([[r/255.0, g/255.0, b/255.0]], dtype=np.float32)

        # 예측
        ret, result, neighbors, dist = knn.findNearest(input_data, k=1)
        pred_label = int(result[0][0])
        label_name = num_to_label

        # 화면에 출력
        cv2.putText(display, f"Predicted : {label_name[pred_label]}", 
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(win_name, display)

    if cv2.waitKey(1) == 27:
        break
    elif cv2.waitKey(1) == ord(' '):
        roi_box = cv2.selectROI(win_name, frame, fromCenter=False, showCrosshair=True)
        if roi_box[2] > 0 and roi_box[3] > 0:
            roi = roi_box
            print(f"ROI 선택됨 : {roi}")

cap.release()
cv2.destroyAllWindows()