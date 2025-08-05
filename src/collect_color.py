import cv2
import csv
import os

# 색상 샘플
label_map = {
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
    cv2.setMouseCallback('Color Collector', mouse_callback, frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    elif key in label_map and click_color is not None:
        label = label_map[key]
        r, g, b = click_color
        print(f"저장됨 : RGB=({r},{g},{b}), Lable={label}")
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([r, g, b, label])
        click_color = None  # 초기화

cap.release()
cv2.destroyAllWindows()