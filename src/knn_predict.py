import csv
import random
import math

samples = []
labels = []

# CSV 파일 불러오기
with open('color_dataset.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        r, g, b , label = row
        label = row[3]
        # RGB 값 정규화 
        samples.append([int(r)/255, int(g)/255, int(b)/255])
        labels.append(label)