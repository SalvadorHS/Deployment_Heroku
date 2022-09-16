from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in  = open("model.pkl","rb")
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Please specify /apidocs in the URL to access the API"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """ Relationship between Corporate Diversification and Financial Performance Measures 
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
            description: The output values
        
    """
    AssetTurnover = request.args.get("AssetTurnover")
    Debt          = request.args.get("Debt")
    QuickRatio    = request.args.get("QuickRatio")
    CashHoldings  = request.args.get("CashHoldings")
    ROA           = request.args.get("ROA")
    prediction    = classifier.predict([[AssetTurnover,Debt,QuickRatio,CashHoldings,ROA]])
    print(prediction)
    return "The predicted value is "+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Please provide a dataset   
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
    
    
