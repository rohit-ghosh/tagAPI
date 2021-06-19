# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 11:33:22 2021

@author: rohit
"""

import uvicorn
from fastapi import FastAPI
from Feedbacks import Feedback
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier = pickle.load(pickle_in)

@app.post('/predict')
def predict_tag(data:Feedback):
    data = data.dict()
    feedback = data['feedback']
    prediction = classifier.predict([feedback])
    return {
        'prediction': prediction[0]
        }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1',port=8000)