import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

#read data
vectorizer = CountVectorizer()
corpus = ["góp gió gặt bão",
          "có làm mới có ăn",
          "đất lành chim đậu",
          "ăn cháo đá bát",
          "gậy ông đập lưng ông",
          "qua cầu rút ván"]
X = vectorizer.fit_transform(corpus)
#print(X.toarray())
y_data = np.array([1, 1, 1, 0, 0, 0])

#test data
x_test = vectorizer.transform(['không làm cạp đất mà ăn']).toarray()

#KNN model
classifier = KNeighborsClassifier(n_neighbors=6)
classifier.fit(X.toarray(), y_data)

y_pred = classifier.kneighbors(x_test)
print(y_pred)
