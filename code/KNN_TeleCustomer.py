import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#read data
df = pd.read_csv('data\\TeleCustomers.csv')

#get data
X = df.drop(['custcat'], axis=1)
y = df['custcat']
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#KNN model from 1 to 40 neighbors
error_rate = []
accuracy_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i, p=2, weights='distance')
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    accuracy_rate.append(accuracy_score(y_test, pred_i))
        

#plot the error rate vs K.Value graph
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, 40), error_rate, color='blue',
         linestyle='dashed', marker='o', markerfacecolor='red',
         markersize=10)
plt.title('Error Rate vs. K.Value')
plt.xlabel('K.Value')
plt.ylabel('Error Rate')
print('Minimum error:', min(error_rate), "at K =", error_rate.index(min(error_rate)))

plt.subplot(1, 2, 2)
plt.plot(range(1, 40), accuracy_rate, color='blue',
         linestyle='dashed', marker='o', markerfacecolor='red',
         markersize=10)
plt.title('Accuracy Rate vs. K.Value')
plt.xlabel('K.Value')
plt.ylabel('Accuracy Rate')
print('Maximum accuracy:', max(error_rate), "at K =", accuracy_rate.index(max(accuracy_rate)))


plt.show()