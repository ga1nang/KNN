import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

#load data
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))

#seperate data
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2)

#KNN model
clf = neighbors.KNeighborsClassifier(n_neighbors=7, p=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 7-NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))

#KNN with distance
clf_distance = neighbors.KNeighborsClassifier(n_neighbors=7, p=2, weights='distance')
clf_distance.fit(X_train, y_train)
y_pred_distance = clf_distance.predict(X_test)

print("Accuracy of 7-NN with major voting and distance weight: %.2f %%" %(100*accuracy_score(y_test, y_pred_distance)))
print(classification_report(y_test, y_pred_distance))

#KNN with custom weight
def myweight(distances):
    sigma2 = .5
    return np.exp(-distances**2/sigma2)

clf_custom = neighbors.KNeighborsClassifier(n_neighbors=7, p=2, weights=myweight)
clf_custom.fit(X_train, y_train)
y_pred_custom = clf_custom.predict(X_test)

print("Accuracy of 7-NN with major voting and custom weight: %.2f %%" %(100*accuracy_score(y_test, y_pred_custom)))
print(classification_report(y_test, y_pred_custom))