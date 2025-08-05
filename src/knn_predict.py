import csv
import random
import math

# CSV 파일 불러오기
data = []
with open('color_dataset.csv', 'r') as f:   # 학습 데이터 로드
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        r, g, b = map(int, row[:3])
        label = row[3]
        data.append(((r/255, g/255, b/255), label)) # RGB 값 정규화