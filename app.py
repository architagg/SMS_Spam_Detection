from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("data/spam.csv" , encoding='latin-1')
    df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
    df.columns=['labels','data']
    df_data = df['data']
    df_label = df['labels']
    corpus = df_data
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    X_train, X_test, y_train, y_test = train_test_split(X, df_label, test_size=0.3, random_state=3)
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        
    return render_template('result.html',prediction = my_prediction)


    




if __name__ == '__main__':
    app.run(debug=True)