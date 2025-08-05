import csv
import cv2
import numpy as np

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

# 텍스트 라벨 -> 숫자 라벨
label_names = sorted(set(labels))
label_to_num = {name : i for i, name in enumerate(label_names)}
num_to_label = {i : name for name, i in label_to_num.items()}
labels_num = [label_to_num[label] for label in labels]

# NumPy 배열로 변환
samples = np.array(samples, dtype=np.float32)
labels_num = np.array(labels_num, dtype=np.int32)

# 학습
train = samples
train_labels = labels_num
test = samples
test_labels = labels_num

# KNN 객체 생성 및 훈련
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

# K 값을 1부터 10까지 변경하며 정확도 테스트
for k in range(1, 11):
    ret, result, neighbors, distance = knn.findNearest(test, k=k)
    correct = np.sum(result.flatten() == test_labels)
    accuracy = correct / result.size * 100.0
    print(f"K={k} : 정확도 = {accuracy : .2f}% ({correct}/{result.size})")