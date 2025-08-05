import cv2
import csv
import os

# 색상 샘플
color_sample = {
    ord('1'): 'red',
    ord('2'): 'blue',
    ord('3'): 'green',
    ord('4'): 'yellow',
    ord('5'): 'black',
    ord('6'): 'white',
    ord('7'): 'gray',
}

# 저장할 파일 경로
csv_path = '../color_dataset.csv'

# 파일 존재하지 않으면 헤더 추가
if not os.path.exists(csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['R', 'G', 'B', 'label'])

# 마우스 이벤트 콜백 함수
click_color = None
def mouse_callback(event, x, y, flags, param):
    global click_color
    if event == cv2.EVENT_LBUTTONDOWN:
        frame = param
        b, g, r = frame[y, x]
        click_color = (r, g, b)  # RGB로 저장

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