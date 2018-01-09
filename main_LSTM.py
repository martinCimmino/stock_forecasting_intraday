#import tensorflow as tf
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))

import utils as UTILS
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np


import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Activation

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

limit = '2018-01-09 13:60' # last sample must be escluded and used for evaluation

# Splitting labels and training
y = data_train.label
X = data_train.drop('label', axis=1)

# Normalization features (Gaussian)
X = UTILS.normalize(X)

X_val = X[limit:] # must be last record
y_val = y[limit:]

print X_val.shape
X_val = X_val.values
X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

X_arr = X[:limit].values # Dataframe to numpy array
y_arr = y[:limit].values

X_arr = X_arr.astype('float32')

print X.shape
print y.shape

#data_val = data['2017-01-01':]
#y_val = data_val.target
#X_val = data_val.drop('target', axis=1)
#X_val = normalize(X_val)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X_arr, y_arr, test_size=1.0/6, shuffle =False ) # shufflle??

#Reshape input to be [samples, time steps, features]
step_size = 1
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# design network
model = Sequential()

print X_train.shape[1]
print  X_train.shape[2]

model.add(LSTM(50, input_shape=(1, 23)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train , y_train, epochs=50, batch_size=5, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()


y = model.predict(X_val) # reshape x val
cls = model.predict_classes(X_val)
print y_val, cls
print y

print("--- %s seconds ---" % (time.time() - start_time))