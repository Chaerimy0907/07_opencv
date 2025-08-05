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
import numpy as np
import matplotlib.pyplot as plt

# 웹캠 연결
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("웹캠 연결 안 됨")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) == 27:    #ESC 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()