import pickle
import string
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
import sys
import pandas as pd

#load model

def load_model(filename):

    model = pickle.load(open(filename,'rb'))

    return model


def load_vectorizer(filename):

    vectorize = pickle.load(open(filename,'rb'))

    return vectorize

def predict_fakenews(model,data):

    predict = model.predict(data)

    return predict

def remove_punct(txt):
    text_remove_punct = [c for c in txt if c not in string.punctuation]
    return text_remove_punct


def regular_expr(txt):
    txt = re.sub('[^a-zA-Z]',' ',str(txt))
    #txt = txt.split()
    return txt


def lemmatiz(txt):
    lemmatizer = WordNetLemmatizer()
    txt = lemmatizer.lemmatize(txt)
    #txt = txt.split()
    return txt


def main():

   filename_model = './model/fake_news_nb.pkl'
   model = load_model(filename_model)

   filename_vectorizer = './model/vectorizer.pkl'
   vectorizer = load_vectorizer(filename_vectorizer)

   truenews = pd.read_csv('./data/True.csv') 
   #truenews.head()
   fakenews = pd.read_csv('./data/Fake.csv') 
   #fakenews.head()

   truenews = truenews.drop([ "subject","date"], axis = 1)
   fakenews = fakenews.drop([ "subject","date"], axis = 1)
   truenews['result'] = 0
   fakenews['result'] = 1
   frames = [truenews, fakenews]
   result = pd.concat(frames)

   
   msk = np.random.rand(len(result)) < 0.8
   train = result[msk].copy()
   test=result[~msk].copy()

   
   test['text'] = test['text'].map(lambda x:  x.lower())
   test['text'] = test['text'].map(lambda x:  regular_expr(x))
   test['text'] = test['text'].map(lambda x:  lemmatiz(x))

   x_test = vectorizer.fit_transform(test['text'])

   y_test = test['result'].to_numpy()

   predict = model.predict(x_test)

   

   count = 0
   for i in range(len(y_test)):
      if y_test[i] == predict[i]:
         count+=1
   k = count/len(y_test)
   print(f'Accuracy : {k}')


if __name__ == "__main__":
    sys.exit(main())

