
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

df = pd.read_csv('Financial_Mexican_Firms.csv', usecols = ['Shannon','AssetTurnover','Debt','QuickRatio','CashHoldings','ROA'])

### Train & Test Split
X  = df.iloc[:, 1:6]
y  = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0) 
df_train = pd.concat([X_train, y_train], axis = 1)

### Handling missing values
from sklearn.impute import KNNImputer

knn       = KNNImputer(n_neighbors = 3)

df_train  = pd.DataFrame(knn.fit_transform(df_train), columns = ['AssetTurnover',
                                                                 'Debt',
                                                                 'QuickRatio',
                                                                 'CashHoldings',
                                                                 'ROA',
                                                                 'Shannon'])

X_train  = df_train.iloc[:, 0:5]
y_train  = df_train.iloc[:, -1]

### Specify the optimal model
optimal_model = RandomForestRegressor(n_estimators      = 118,
                                      min_samples_split = 2,
                                      min_samples_leaf  = 4,
                                      max_features      = 'auto',
                                      max_depth         = 25).fit(X_train, y_train)




### Create a Pickle file (serialization)
import pickle
pickle_out = open("model.pkl","wb")
pickle.dump(optimal_model, pickle_out)
pickle_out.close()
