#import tensorflow as tf
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))

import utils as UTILS
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

start_time = time.time()

from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries

#import requests
#r = requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=MSFT&interval=1min&apikey=NRE9GCFWLFXJWJHT')
#print r.__dict__

apikey = 'NRE9GCFWLFXJWJHT'
t_period = '5min'
stock = 'AAPL'
series_type = 'open'

ts = TimeSeries(key=apikey, output_format='pandas')
data, meta_data = ts.get_intraday(symbol=stock,interval=t_period)
#print data.shape
#print data.columns.values

# datetime have differrent format
for idx, el in enumerate(data.index._data):
    data.index._data[idx] = el[:-3]
data["trend"] = data["4. close"].subtract(data["1. open"], fill_value=0)
del data["4. close"]
del data["1. open"]

print data.shape

ti = TechIndicators(key=apikey, output_format='pandas')
#1
bbands, meta_data_bbands = ti.get_bbands(symbol=stock, interval=t_period, series_type=series_type)
print bbands.shape
#2
ema, meta_data_ema = ti.get_ema(symbol=stock, interval=t_period, series_type= series_type)
print ema.shape
#3
wma, meta_data_wma = ti.get_wma(symbol=stock, interval=t_period, series_type= series_type)
print wma.shape
#4
sma, meta_data_sma = ti.get_sma(symbol=stock, interval=t_period, series_type= series_type)
print sma.shape
#5
rsi, meta_data_rsi = ti.get_rsi(symbol=stock, interval=t_period, series_type= series_type)
print rsi.shape
#6
macd, meta_data_macd = ti.get_macd(symbol=stock,interval=t_period, series_type= series_type)
print macd.shape
#7
stoch, meta_data_stoch = ti.get_stoch(symbol=stock, interval=t_period)
print stoch.shape
#8
adx, meta_data_adx = ti.get_adx(symbol=stock, interval=t_period)
print adx.shape
#9
cci, meta_data_cci = ti.get_cci(symbol=stock, interval=t_period)
print cci.shape
#10
aroon, meta_data_aroon = ti.get_aroon(symbol=stock, interval=t_period, series_type=series_type)
print aroon.shape
#11
ad, meta_data_ad = ti.get_ad(symbol=stock, interval=t_period)
print ad.shape
#12
obv, meta_data_obv = ti.get_obv(symbol=stock, interval=t_period)
print obv.shape
#13
mom, meta_data_mom = ti.get_mom(symbol=stock, interval=t_period, series_type= series_type)
#14
willr, meta_data_willr = ti.get_willr(symbol=stock, interval=t_period)


result = data.join(bbands, how='left').join(ema , how='left').join( wma, how='left').join(sma , how='left').join(rsi , how='left').join(macd , how='left').join(stoch , how='left').join( adx, how='left').join(cci , how='left').join(aroon , how='left').join(ad, how='left').join(obv,how='left').join(mom,how='left').join(willr,how='left')   # join at the end
result['label'] = np.where(result['trend']>= 0, 1, 0) # 1 price-up and 0 price-down
del result["trend"]
#print result
print result.columns.values

#result.tail(10) # number of rows to select
#result.describe(include='all')
#ax = data['trend'].plot(figsize=(9, 5))
#ax.set_ylabel("Price ($)")
#ax.set_xlabel("Time")
#plt.show()

# Balancing labels
data_train = UTILS.rebalance(result)

data_val = data_train['2018-01-05 16:00':] # must be last record
y_val = data_val.label
X_val = data_val.drop('label', axis = 1)
X_val = UTILS.normalize(X_val)

# Splitting labels and training
y = data_train.label
X = data_train.drop('label', axis=1)

# Normalization features (Gaussian)
X = UTILS.normalize(X)

X_arr = X.values # Dataframe to numpy array
y_arr = y.values

print X.shape
print y.shape

#data_val = data['2017-01-01':]
#y_val = data_val.target
#X_val = data_val.drop('target', axis=1)
#X_val = normalize(X_val)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=1.0/6, shuffle =False ) # shufflle??

models = [GaussianNB(),
          SVC(random_state=5),
          RandomForestClassifier(random_state=5),
          MLPClassifier(random_state=5)]

for model in models:
    model.fit(X_train, y_train)

UTILS.scores(models, X_test, y_test)

#print models[0].estimator.get_params().keys()

'''
# Grid search for each model
grid_data = [ {'kernel': ['rbf', 'sigmoid'],
               'C': [0.1, 1, 10, 100],
               'random_state': [5]},
              {'n_estimators': [10, 50, 100],
               'criterion': ['gini', 'entropy'],
               'max_depth': [None, 10, 50, 100],
               'min_samples_split': [2, 5, 10],
               'random_state': [5]},
              {'hidden_layer_sizes': [10, 50, 100],
               'activation': ['identity', 'logistic', 'tanh', 'relu'],
               'solver': ['lbfgs', 'sgd', 'adam'],
               'learning_rate': ['constant', 'invscaling', 'adaptive'],
               'max_iter': [200, 400, 800],'random_state': [5]}]

models_grid = list()

models = models[1:4] # not searching on GaussianNB

for i in range(3):
    print models[i].get_params().keys()
    grid = GridSearchCV(models[i], grid_data[i], scoring='accuracy').fit(X_train, y_train) # scoring='f1'
    print("Best setup: {}" .format(grid.best_params_))
    model = grid.best_estimator_ # select best one
    models_grid.append(model)

print ('Grid Search')
UTILS.scores(models_grid, X_test, y_test)
'''

rf_model = models[2]
y_pred = rf_model.predict(X_val)
print ('predicted: {}'.format(y_pred))
print y_val

print("--- %s seconds ---" % (time.time() - start_time))
