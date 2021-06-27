import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer #Import Count Vectorizer
import numpy as np
import xgboost as xgb
import re
from joblib import dump
from joblib import load

here = os.path.dirname(__file__)
modelpath = os.path.join(here, 'model.joblib.dat')
vocabpath = os.path.join(here, 'vocab.joblib.dat')
vocab = load(vocabpath)


cv = CountVectorizer(min_df=0, lowercase=True,vocabulary = vocab)


loaded_model = load(modelpath)
print("Loaded model")
def remove_pattern(input_txt, pattern):
  r = re.findall(pattern, input_txt)
  for i in r:
    input_txt = re.sub(i, '', input_txt)
  return input_txt


def load_predict(input_text):
    text = np.vectorize(remove_pattern)(input_text, "@[\w]*")
    #tfid.fit(text)
    print("pass1")
    pred_text = cv.transform(text)
    print("pass2")
    return (loaded_model.predict(pred_text))
    print("pass3")
    
    





