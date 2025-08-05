import cv2
import csv
import os

# 웹캠 연결
cap = cv2.VideoCapture(0)
cv2.namedWindow('Color Collector')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Color Collector', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()