import numpy as np
import pandas as pd
import os
import nltk
import re
nltk.download('stopwords')
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
app=Flask(__name__)
model=load_model("weightsone.h5")
with open('count_vectorizer.pkl', 'rb') as file:
    count = pickle.load(file)
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        testdata=request.form['name']
        stemming = PorterStemmer()
        testdata = re.sub('[^a-zA-Z]', ' ', testdata)
        testdata = testdata.lower()
        testdata = testdata.split()
        testdata = [word for word in testdata if word not in set(stopwords.words('english'))]
        testdata = [stemming.stem(word) for word in testdata]
        testdata = ' '.join(testdata)
        testdataone = count.transform([testdata]).toarray()
        prediction = model.predict(testdataone)
        if prediction > 0.5:
            text='Positive'
        else:
            text='Negative'
    return render_template("result.html",output=text)
if __name__=='__main__':
    app.run()