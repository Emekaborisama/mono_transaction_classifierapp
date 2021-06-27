from flask import Flask, request, jsonify, Response
from flask import render_template, url_for, redirect, flash
#from spacy import displacy
from app.predict import load_predict
from flask_cors import CORS, cross_origin



app = Flask(__name__)
cors = CORS(app)


@app.route("/")
def home():
    return("welcome to monosampleapi")

@app.route("/mono_transapi", methods=['POST'])
def pred():
    data = request.form['text']
    cat = load_predict([data])
    return(cat[0])
    



