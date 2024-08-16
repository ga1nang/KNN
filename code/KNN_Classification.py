import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# file = open("data\\Iris.csv", "r")
# data = csv.reader(file)
# data = np.array(list(data))
# data = np.delete(data, 0, 0)
# data = np.delete(data, 0, 1)
# file.close()

# trainingSet = data[:149]
# testingSet = data[149:]

# def computeDistance(dataPoint1, dataPoint2):
#     res = 0
#     for i in range(4):
#         res += (float(dataPoint1[i]) - float(dataPoint2[i]))**2
#     return math.sqrt(res)


# def computeKnearestNeighbor(trainingSet, item, k):
#     distances = []
#     for dataPoint in trainingSet:
#         distances.append(
#             {
#                 "label": dataPoint[-1],
#                 "distances": computeDistance(item, dataPoint)
#             }
#         )
#     distances.sort(key=lambda x: x["distances"])
#     labels = [item["label"] for item in distances]
#     return labels[:k]

# def voteTheDistances(array):
#     labels = set(array)
#     res = ""
#     maxOccur = 0
#     for label in labels:
#         num = array.count(label)
#         if (num > maxOccur):
#             maxOccur = num
#             res = label
            
#     return res


# k = 5
# for item in testingSet:
#     knn = computeKnearestNeighbor(trainingSet, item, k)
#     res = voteTheDistances(knn)
#     print("GT = ", item[-1], ", Prediction: =", res)
    
    
#######How to select K in K-NN
url = "data\\Iris.csv"
names = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    
dataset = pd.read_csv(url, names=names)

X = dataset.iloc[1:, :-1].values
y = dataset.iloc[1:, 4].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_pred != y_test)

print("Accuracy score: ", round(accuracy_score(y_test, y_pred), 2))

print(classification_report(y_test, y_pred))

#plot graph
error = []
for i in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(accuracy_score(pred_i, y_test))
    
plt.figure(figsize=(12, 5))
plt.plot(range(1, 30), error, color='blue', marker='o',
        markerfacecolor='yellow', markersize=10)
plt.title('Accuracy vs K Value')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.show()