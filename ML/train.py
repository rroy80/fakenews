import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling2D, Dropout, Flatten, Dense,BatchNormalization,Activation

nltk.download('stopwords')
set(stopwords.words('english'))

truenews = pd.read_csv('../data/True.csv')
#truenews.head()
fakenews = pd.read_csv('../data/Fake.csv')
#fakenews.head()

truenews = truenews.drop([ "subject","date"], axis = 1)
fakenews = fakenews.drop([ "subject","date"], axis = 1)

truenews['result'] = 0
fakenews['result'] = 1

frames = [truenews, fakenews]

result = pd.concat(frames)


result['title'] = result['title'].map(lambda x:  x.lower())
result['text'] = result['text'].map(lambda x:  x.lower())


msk = np.random.rand(len(result)) < 0.8
train = result[msk].copy()
test=result[~msk].copy()

def remove_punct(txt):
    text_remove_punct = [c for c in txt if c not in string.punctuation]
    return text_remove_punct

def regular_expr(txt):
    txt = re.sub('[^a-zA-Z]',' ',str(txt))
    #txt = txt.split()
    return txt

train['text'] = train['text'].map(lambda x:  regular_expr(x))
test['text'] = test['text'].map(lambda x:  regular_expr(x))


lemmatizer = WordNetLemmatizer()

def lemmatiz(txt):
    txt = lemmatizer.lemmatize(txt)
    #txt = txt.split()
    return txt

train['text'] = train['text'].map(lambda x:  lemmatiz(x))
test['text'] = test['text'].map(lambda x:  lemmatiz(x))


def regular_token(txt):
    #txt = re.sub('[^a-zA-Z]',' ',str(txt))
    txt = txt.split()
    return txt

num_features=4900
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features,lowercase = False, ngram_range = (1,2))
x_train = tfidf.fit_transform(train['text'])
x_test = tfidf.transform(test['text'])


pickle.dump(tfidf, open('../model/vectorizer.pkl','wb'))


clf = MultinomialNB()
y_train = train['result'].to_numpy()
clf.fit(x_train,y_train)
predict_test = clf.predict(x_test)


y_test = test['result'].to_numpy()

count = 0
for i in range(len(y_test)):
    if y_test[i] == predict_test[i]:
        count+=1
k = count/len(y_test)
print(f'Accuracy : {k}')

predict = clf.predict(x_train)

count = 0
for i in range(len(y_train)):
    if y_train[i] == predict[i]:
        count+=1
k = count/len(y_train)
print(f'Accuracy : {k}')



pickle.dump(clf, open('../model/fake_news_nb.pkl','wb'))


cm=confusion_matrix(y_train, predict)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()


######Training Accuracy
def accuracy_score(y_train,predict):
  count = 0
  True_Positive = 0
  True_Negative = 0
  False_Positive = 0
  False_Negative = 0
  for i in range(len(y_train)):
    if y_train[i] == predict[i]:
        if predict[i] == 1:
            True_Positive += 1
        else:
            True_Negative += 1
    else:
        if predict[i] == 1:
            False_Positive += 1
        else:
            False_Negative += 1
  precision = True_Positive/(True_Positive + False_Positive)
  recall = True_Positive/(True_Positive + False_Negative)
  f1 = 2 * precision * recall / (precision + recall)
  return (precision, recall, f1)




precision,recall,f1 = accuracy_score(y_train, predict)

print(f'Training Accuracy Naive Bayes Precision : {precision} Recall : {recall}, f1 : {f1}')

precision,recall,f1 = accuracy_score(y_test, predict_test)

print(f'Testing Accuracy Naive Bayes Precision : {precision} Recall : {recall}, f1 : {f1}')


y_train = train['result'].to_numpy()
mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=100)
x_train_array_1=x_train.toarray()
x_test_array_1=x_test.toarray()

mlp.fit(x_train_array_1, y_train)
predict_MLP_train=mlp.predict(x_train_array_1)

count = 0
for i in range(len(y_train)):
    if y_train[i] == predict_MLP_train[i]:
        count+=1
k = count/len(y_train)
print(f'Accuracy : {k}')

predict = mlp.predict(x_test.toarray())


count = 0
for i in range(len(y_test)):
    if y_test[i] == predict_test[i]:
        count+=1
k = count/len(y_test)
print(f'Accuracy : {k}')

cm=confusion_matrix(y_train, predict_MLP_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()


precision,recall,f1 = accuracy_score(y_train, predict_MLP_train)

print(f'Training Accuracy NN Precision : {precision} Recall : {recall}, f1 : {f1}')

precision,recall,f1 = accuracy_score(y_test, predict)

print(f'Testing Accuracy NN Precision : {precision} Recall : {recall}, f1 : {f1}')


x_train_array = x_train.toarray().reshape(-1,70, 70,1)
print(x_train_array.shape)
x_test_array = x_test.toarray().reshape(-1,70, 70,1)
y_train
print(y_train.shape)


clf = GaussianNB()
clf.fit(x_train_array_1, y_train)
predict_gaussian = clf.predict(x_train_array_1)

count = 0
for i in range(len(y_train)):
    if y_train[i] == predict_gaussian[i]:
        count+=1
k = count/len(y_train)
print(f'Accuracy : {k}')

predict_gaussian_test = clf.predict(x_test_array_1)

count = 0
for i in range(len(y_test)):
    if y_test[i] == predict_gaussian_test[i]:
        count+=1
k = count/len(y_test)
print(f'Accuracy : {k}')

cm=confusion_matrix(y_train, predict_gaussian)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()

precision,recall,f1 = accuracy_score(y_train, predict_gaussian)

print(f'Training Accuracy Gaussian Precision : {precision} Recall : {recall}, f1 : {f1}')

precision,recall,f1 = accuracy_score(y_test, predict_gaussian_test)

print(f'Testing Accuracy Gaussian Precision : {precision} Recall : {recall}, f1 : {f1}')

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train_array_1, y_train)
predict_neigh_train = neigh.predict(x_train_array_1)

count = 0
for i in range(len(y_train)):
    if y_train[i] == predict_neigh_train[i]:
        count+=1
k = count/len(y_train)
print(f'Accuracy : {k}')

predict_neigh_test = neigh.predict(x_test_array_1)

count = 0
for i in range(len(y_test)):
    if y_test[i] == predict_neigh_test[i]:
        count+=1
k = count/len(y_test)
print(f'Accuracy : {k}')

cm=confusion_matrix(y_train, predict_neigh_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()

precision,recall,f1 = accuracy_score(y_train, predict_neigh_train)

print(f'Training Accuracy KNN Precision : {precision} Recall : {recall}, f1 : {f1}')

precision,recall,f1 = accuracy_score(y_test, predict_neigh_test)

print(f'Testing Accuracy KNN Precision : {precision} Recall : {recall}, f1 : {f1}')


clf = LogisticRegression(random_state=0).fit(x_train_array_1, y_train)
Predict_logitsic = clf.predict(x_train_array_1)

count = 0
for i in range(len(y_train)):
    if y_train[i] == Predict_logitsic[i]:
        count+=1
k = count/len(y_train)
print(f'Accuracy : {k}')

predict_logitsic_test = clf.predict(x_test_array_1)

count = 0
for i in range(len(y_test)):
    if y_test[i] == predict_logitsic_test[i]:
        count+=1
k = count/len(y_test)
print(f'Accuracy : {k}')

cm=confusion_matrix(y_train, Predict_logitsic)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()

precision,recall,f1 = accuracy_score(y_train, Predict_logitsic)

print(f'Training Accuracy Logistic Regression Precision : {precision} Recall : {recall}, f1 : {f1}')

precision,recall,f1 = accuracy_score(y_test, predict_logitsic_test)

print(f'Testing Accuracy Logistic Regression Precision : {precision} Recall : {recall}, f1 : {f1}')



pca = PCA(n_components=2)

X_train = pca.fit_transform(x_train_array_1)
X_test = pca.transform(x_test_array_1)


cls = svm.SVC(kernel="linear")
y_train = train['result'].to_numpy()
cls.fit(X_train, y_train)


predict_SVM_train = cls.predict(X_train)

count = 0
for i in range(len(y_train)):
    if y_train[i] == predict_SVM_train[i]:
        count+=1
k = count/len(y_train)
print(f'Accuracy : {k}')

predict_SVM_test = cls.predict(X_test)

count = 0
for i in range(len(y_test)):
    if y_test[i] == predict_SVM_test[i]:
        count+=1
k = count/len(y_test)
print(f'Accuracy : {k}')

cm=confusion_matrix(y_train, predict_SVM_train)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=clf.classes_)
disp.plot()
plt.show()


precision,recall,f1 = accuracy_score(y_train, predict_SVM_train)

print(f'Training Accuracy SVM : {precision} Recall : {recall}, f1 : {f1}')

precision,recall,f1 = accuracy_score(y_test, predict_SVM_test)

print(f'Testing Accuracy SVM Precision : {precision} Recall : {recall}, f1 : {f1}')

