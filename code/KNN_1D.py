import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

#read data
data = pd.read_csv('data\iris_1D.csv')

#get data
x_data = data['Petal_Length'].to_numpy()
x_data = x_data.reshape(6, 1)
y_data = data['Label'].to_numpy()

#test data
x_test = [[2.4]]

#KNN with 1 neighbors
classifier1 = KNeighborsClassifier(n_neighbors=1)
classifier1.fit(x_data, y_data)

y_pred1 = classifier1.predict(x_test)
print(y_pred1)

#KNN with 3 neighbors
classifier2 = KNeighborsClassifier(n_neighbors=3)
classifier2.fit(x_data, y_data)

y_pred2 = classifier2.predict(x_test)
print(y_pred2)