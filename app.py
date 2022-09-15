# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 2022

@author: Salvador HS
"""

import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image

pickle_in  = open("random_forest_diversification.pkl","rb")
classifier = pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
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
   
    prediction = classifier.predict([[AssetTurnover,Debt,QuickRatio,CashHoldings, ROA]])
    print(prediction)
    return prediction


def main():
    st.title("Corporate Diversification")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Corporate Diversification and Financial performance Measures</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    AssetTurnover = st.text_input("AssetTurnover","Type Here")
    Debt          = st.text_input("Debt","Type Here")
    QuickRatio    = st.text_input("QuickRatio","Type Here")
    CashHoldings  = st.text_input("CashHoldings","Type Here")
    ROA           = st.text_input("ROA","Type Here")

    result=""
    if st.button("Predict"):
        result = predict_corporate_diversification(AssetTurnover, Debt, QuickRatio, CashHoldings, ROA)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("This application shows the relationship between Corporate Diversification and Financial Performance")
        st.text("Financial Performance prediction based on Corporate Diversification are built in a different app")
        st.text("These data patterns are found by using a Random Forest Regressor.")

if __name__=='__main__':
    main()
    
    
