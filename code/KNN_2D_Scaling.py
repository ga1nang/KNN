import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#read data
data = pd.read_csv('data\iris_2D_mm.csv')

#get data
x_data = data[['Petal_Length', 'Petal_Width']].to_numpy()
x_data = x_data.reshape(6, 2)
y_data = data['Label'].to_numpy()

#scale data
scaler = StandardScaler()
scaler.fit(x_data)
x_data = scaler.transform(x_data)
#print(x_data)

#test_data
x_test = [[2.6, 7.0]]
x_test = scaler.transform(x_test)
#print(x_test)

#KNN with 6 neighbors
classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(x_data, y_data)
y_pred = classifier.predict(x_test)
print(y_pred)