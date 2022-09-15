# -*- coding: utf-8 -*-
"""
@author: Salvador HS
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in  = open("random_forest_diversification.pkl","rb")
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_corporate_diversification(AssetTurnover, Debt, QuickRatio, CashHoldings, ROA):
    """Relationship between Corporate Diversification and Financial Performance Measures
    The data patterns are found by using a Random Forest Regressor.
    ---
    parameters:  
      - name: AssetTurnover
        in: query
        type: number
        required: true
      - name: Debt
        in: query
        type: number
        required: true
      - name: QuickRatio
        in: query
        type: number
        required: true
      - name: CashHoldings
        in: query
        type: number
        required: true
      - name: ROA
        in: query
        type: number
        required: true
    responses:
        200:
            description: The diversification value using entropy is 
        
    """
   
    prediction = classifier.predict([[AssetTurnover ,Debt, QuickRatio, CashHoldings, ROA]])
    print(prediction)
    return "The prediction is "+str(prediction)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    
    