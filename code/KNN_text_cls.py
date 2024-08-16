import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

#prepare data
corpus = [
    "góp gió gặt bão",
    "có làm mới có ăn",
    "đất lành chim đậu",
    "ăn cháo đá bát",
    "gậy ông đập lưng ông",
    "qua cầu rút ván"
]

n_doc = len(corpus)

labels = [1, 1, 1, 0, 0, 0]

cate_2_label = {
    "positive": 1,
    "negative": 0
}
X = np.array(corpus)
y = np.array(labels)

#label to categories
def label_2_cate(labels):
    key_list = list(cate_2_label.keys())
    val_list = list(cate_2_label.values())
    
    position = [val_list.index(label) for label in labels]
    return np.array(key_list)[position]


#convert text to vector using TF-IDF transform
def calculate_tfidf(X_vectorized):
    tf = np.log(X_vectorized + 1)
    df = np.sum(X_vectorized, axis = 0)
    idf = np.log((n_doc + 1) / (df + 1)) + 1
    tfidf = tf * idf
    
    return idf, tf, tfidf

def compute_norm(tfidf_vec):
    norm = np.linalg.norm(tfidf_vec, axis = 1)
    n_doc = tfidf_vec.shape[0]
    for i in range(n_doc):
        tfidf_vec[i] /= norm[i]
        
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X).toarray()
print("Vocab: ", vectorizer.get_feature_names_out())

X_idf, x_tf, X_tfidf = calculate_tfidf(X_vectorized)

#normalize TF-IDF values by L2 norm
compute_norm(X_tfidf)

#train model KNN with 1 neighbor
knn_cls = KNeighborsClassifier(n_neighbors=3)
knn_cls.fit(X_tfidf, y)
preds = knn_cls.predict(X_tfidf)
print(preds)

#using pipeline of sklearn
text_clf_model = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', KNeighborsClassifier(n_neighbors=1)),
                           ])

text_clf_model.fit(X, y)

preds = text_clf_model.predict(X)
print(preds)

#inference
test_text = np.array(["không làm cạp đất mà ăn"])
test_vec = vectorizer.transform(test_text).toarray()
test_tf = np.log(test_vec + 1)
test_tfidf = test_tf * X_idf
compute_norm(test_tfidf)
pred = knn_cls.predict(test_tfidf)
print(label_2_cate(pred))