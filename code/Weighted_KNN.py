from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np

#read data
dataset = datasets.load_iris()
X = dataset.data
y = dataset.target

#seperate data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


neighbors = 6


#KNN with uniform weight
classifier_uniform = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform', p=2)
classifier_uniform.fit(X_train, y_train)
y_pred = classifier_uniform.predict(X_test)

print("Knn with Uniform weight")
print(f"Accuracy %.2f%%"%(100*accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))

#KNN with distance weight
classifier_distance = KNeighborsClassifier(n_neighbors=neighbors, weights='distance', p=2)
classifier_distance.fit(X_train, y_train)
y_pred = classifier_distance.predict(X_test)

print("Knn with distance weight")
print(f"Accuracy %.2f%%"%(100*accuracy_score(y_test, y_pred)))
print(classification_report(y_test,y_pred))

#KNN with customize weight
def customizeWeight(distances):
    sigma = 0.5
    return np.exp(-distances**2/sigma)

classifier_custom = KNeighborsClassifier(n_neighbors=neighbors, weights=customizeWeight, p=2)
classifier_custom.fit(X_train, y_train)
y_pred = classifier_custom.predict(X_test)

print("Knn with custom weight")
print(f"Accuracy %.2f%%"%(100*accuracy_score(y_test, y_pred)))
print(classification_report(y_test, y_pred))
