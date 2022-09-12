import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_csv('hiring.csv')

# Handling NaN
df['experience'].fillna(0, inplace=True)
df['test_score'].fillna(df['test_score'].mean(), inplace=True)

# Option 1: Create a Dictionary to map
dictionary1 = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}

# Option 2: Define a function that returns a dictionary 
def encoder(word):
    dictionary2 = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return dictionary2[word]

df['experience'] = df['experience'].map(dictionary1)            # Option 1
#df['experience'] = df['experience'].apply(lambda x: encoder(x)) # Option 2

X = df.iloc[:, :3]
y = df.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))