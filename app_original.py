"""
Project: Kintsugi

Enter text to predict depression
"""

#Import all required Packages
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request,render_template
import string
import os


#Intialize Flask application
app = Flask(__name__)
templates_dir = os.path.join(app.root_path, 'templates')

@app.route('/')
#Home page for the Application
def home():
    return render_template('home.html')

#Route for prediction template
@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

#Route for Data Preview of Working Model
@app.route('/data_preview',methods=['GET', 'POST'])
def data_preview():
    return render_template('data_preview.html')
    
#Route for html data table template
@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/k_home')
def k_home():
    return render_template('k_home.html')

@app.route('/depression_type')
def depression_type():
    return render_template('depression_type.html')

@app.route('/k_about')
def k_about():
    return render_template('k_about.html')

@app.route('/about')
def about():
    return render_template('about.html')

#Prediction
@app.route("/predict",methods=['POST'])
def predict():
    #Load Model and Dependencies
    enc = pickle.load(open("encoder.pickle", "rb"))
    model = pickle.load(open("model.pickle", "rb"))
    tweet = request.form['desc']
    list_tweet=[tweet]
    vect_tweet=enc.transform(list_tweet)
    output=model.predict(vect_tweet)  
    return render_template('prediction.html',thought=tweet, result="Ooh Sorry, You're Depressed!! We will get through this together:)" if output==0 else "Cheers, You're Not Depressed!!", count = int(1))
    
# Clear Browser Cache
def before_request():
    app.jinja_env.cache = {}


if __name__=='__main__':
    app.before_request(before_request)
    app.run()