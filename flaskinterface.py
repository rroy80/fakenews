import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('./model/fake_news_nb.pkl', 'rb'))
vectorizer = pickle.load(open('./model/fake_news_nb.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    test['text'] = test['text'].map(lambda x:  x.lower())
    test['text'] = test['text'].map(lambda x:  regular_expr(x))
    test['text'] = test['text'].map(lambda x:  lemmatiz(x))
    x_test = vectorizer.fit_transform(test['text'])
    predict = model.predict(x_test)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(predict))

@app.route('/results',methods=['POST'])

if __name__ == "__main__":
    app.run(debug=True)