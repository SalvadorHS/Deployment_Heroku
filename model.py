import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
warnings.filterwarnings('ignore')
df = pd.read_csv("Financial_Mexican_Firms.csv", usecols = ['Shannon','AssetTurnover','Debt','QuickRatio','CashHoldings','ROA'])

### Train & Test Split

X  = df.iloc[:, 1:6]
y  = df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0) 
df_train = pd.concat([X_train, y_train], axis = 1)
df_test  = pd.concat([X_test, y_test], axis = 1)

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

### Handling outliers
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

# Specify outlier model detectors for training data 
SVM_detector_train = OneClassSVM(kernel = 'rbf', gamma = 0.05, nu = 0.1).fit(df_train.values) 
kNN_detector_train = NearestNeighbors(n_neighbors = 2).fit(df_train.values)
IRF_detector_train = IsolationForest(contamination = .05).fit(df_train.values)

# Predict 
SVM_predictions    = SVM_detector_train.predict(df_train.values)
distances, indexes = kNN_detector_train.kneighbors(df_train.values)
IRF_predictions    = IRF_detector_train.predict(df_train.values)

# Filter anomalies
SVM_outlier_values = df_train[(SVM_predictions < 0)]
kNN_outlier_values = df_train[distances.mean(axis = 1) > 0.8]
IRF_outlier_values = df_train[(IRF_predictions < 0)]

# Plot distances in kNN
plt.subplot(1, 2, 1)
plt.plot(distances.mean(axis =1))
plt.ylabel('kNN Distances')

# Plot anomalies
plt.subplot(1, 2, 2)
plt.scatter(df_train['ROA'], df_train['Debt'])
plt.scatter(kNN_outlier_values['ROA'], kNN_outlier_values['Debt'], edgecolors = 'r')
plt.show()

### Feature Engineering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

std_scaler = StandardScaler()
mmx_scaler = MinMaxScaler()
rob_scaler = RobustScaler()

X_train = pd.DataFrame(rob_scaler.fit_transform(X_train), columns = ['AssetTurnover',
                                                                 'Debt',
                                                                 'QuickRatio',
                                                                 'CashHoldings',
                                                                 'ROA'])

X_test = pd.DataFrame(rob_scaler.transform(X_test), columns = ['AssetTurnover',
                                                                 'Debt',
                                                                 'QuickRatio',
                                                                 'CashHoldings',
                                                                 'ROA'])

### Hyperparameter Tunning
# Specify different values for the tunning process
kfold             = KFold(n_splits = 5, random_state = None, shuffle = False)

n_estimators      = [int(x) for x in np.linspace(start = 50, stop = 200, num = 12)] 
max_features      = ['auto', 'sqrt'] 
max_depth         = [int(x) for x in np.linspace(5, 30, 6)] 
min_samples_split = [int(x) for x in np.linspace(2, 20, 6)] 
min_samples_leaf  = [int(x) for x in np.linspace(1, 20, 6)] 

#Create parameter grid
random_grid ={'n_estimators'     :n_estimators,
              'max_features'     :max_features,
              'max_depth'        :max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf' :min_samples_leaf}

#Create Random Forest object
rf = RandomForestRegressor()

#Randomized Search CV
rf_search = RandomizedSearchCV(rf, 
                               random_grid, 
                               scoring      = 'r2', 
                               n_iter       = 10, 
                               cv           = 10, 
                               n_jobs       = -1)

### Fit the model and measure time to execute
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        
start_time = timer(None) # timing starts from this point for "start_time" variable
rf_search.fit(X_train,y_train)
timer(start_time) # timing ends here for "start_time" variable

### Get the best tunning parameters
rf_search.best_params_

### Specify the optimal model
optimal_model = RandomForestRegressor(n_estimators      = 118,
                                      min_samples_split = 2,
                                      min_samples_leaf  = 4,
                                      max_features      = 'auto',
                                      max_depth         = 25).fit(X_train, y_train)

### Evaluate Performance
import time
start_time = time.time()

y_pred     = optimal_model.predict(X_test)
    
print('R2  :', r2_score(y_test, y_pred))
print('MAE :', mean_absolute_error(y_test, y_pred))
print('MSE :', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
print('Processing time: %s seconds' % round((time.time() - start_time), 4))

### Deployment File
import pickle
file = open("random_forest_diversification.pkl", 'wb')
pickle.dump(optimal_model, file)

