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

from sklearn.naive_bayes import MultinomialNB

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

truenews.head()

frames = [truenews, fakenews]

result = pd.concat(frames)

result.head()

result['title'] = result['title'].map(lambda x:  x.lower())
result['text'] = result['text'].map(lambda x:  x.lower())


msk = np.random.rand(len(result)) < 0.8
train = result[msk].copy()
test=result[~msk].copy()
print(test)

def remove_punct(txt):
    text_remove_punct = [c for c in txt if c not in string.punctuation]
    return text_remove_punct

def regular_expr(txt):
    txt = re.sub('[^a-zA-Z]',' ',str(txt))
    #txt = txt.split()
    return txt

train['text'] = train['text'].map(lambda x:  regular_expr(x))
test['text'] = test['text'].map(lambda x:  regular_expr(x))

print(test['text'])


lemmatizer = WordNetLemmatizer()

def lemmatiz(txt):
    txt = lemmatizer.lemmatize(txt)
    #txt = txt.split()
    return txt



string.punctuation

def remove_punct(txt):
    text_remove_punct = [c for c in txt if c not in string.punctuation]
    return text_remove_punct


def regular_expr(txt):
    txt = re.sub('[^a-zA-Z]',' ',str(txt))
    #txt = txt.split()
    return txt

train['text'] = train['text'].map(lambda x:  lemmatiz(x))
test['text'] = test['text'].map(lambda x:  lemmatiz(x))

print(train['text'])

def batchpgm(txt):
    #txt = re.sub('[^a-zA-Z]',' ',str(txt))
    txt = txt.split()
    return txt

print(train['text'])



tfidf = TfidfVectorizer(max_features = 125772,lowercase = False, ngram_range = (1,2))
x_train = tfidf.fit_transform(train['text'])
x_test = tfidf.fit_transform(test['text'])


pickle.dump(tfidf, open('../model/vectorizer.pkl','wb'))
print(x_test.shape)
print(x_train.shape)

print(result.count)


clf = MultinomialNB()
y_train = train['result'].to_numpy()
clf.fit(x_train,y_train)
predict_test = clf.predict(x_test)

print(type(y_train))

y_test = test['result'].to_numpy()

print(type(y_test))

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


######Training Accuarcy

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

print(f' Training Precision : {precision}')
print(f'Training Recall : {recall}')
print(f'Training F1 : {2 * precision * recall /(precision + recall)}')



##Test Accuracy


count = 0
True_Positive = 0
True_Negative = 0
False_Positive = 0
False_Negative = 0
for i in range(len(y_test)):
    if y_test[i] == predict_test[i]:
        if predict_test[i] == 1:
            True_Positive += 1
        else:
            True_Negative += 1
    else:
        if predict_test[i] == 1:
            False_Positive += 1
        else:
            False_Negative += 1

precision = True_Positive/(True_Positive + False_Positive)
recall = True_Positive/(True_Positive + False_Negative)

print(f' Testing Precision : {precision}')
print(f'Testing Recall : {recall}')
print(f'Testing F1 : {2 * precision * recall /(precision + recall)}')



