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

cap.release()
cv2.destroyAllWindows()