import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import pandas as pd
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
import pickle

style.use('ggplot')
mpl.use('TkAgg')

df = pdr.get_data_yahoo('AAPL')

# data.Close.plot()
#
# plt.show()
# plt.gcf().clear()

# data = data[['Adj Open', 'Adj High', 'Adj Low', 'Adj Close', 'Adj Volume']]


ratio = df['Adj Close'] / df['Close']
df['Adj Open'] = df['Open'] * ratio
df['Adj High'] = df['High'] * ratio
df['Adj Low'] = df['Low'] * ratio
df['Adj Volume'] = df['Volume'] / ratio

df['HL_PCT'] = (df['Adj High'] - df['Adj Close']) / df['Adj Close'] * 100.0
df['PCT_change'] = (df['Adj Close'] - df['Adj Open']) / df['Adj Open'] * 100.0

df = df[['Adj Close', 'Adj Volume', 'HL_PCT', 'PCT_change']]

forecastCol = 'Adj Close'

df.fillna(-9999999, inplace=True)

forecastOut = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecastCol].shift(-forecastOut)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
XLately = X[-forecastOut:]
X = X[:-forecastOut]

df.dropna(inplace=True)

y = np.array(df['label'])

XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(XTrain, yTrain)

accuracy = clf.score(XTest, yTest)

forecastSet = clf.predict(XLately)

print(forecastSet, (accuracy*100), forecastOut)

df['Forecast'] = np.nan

lastDate = df.iloc[-1].name
lastUnix = lastDate.timestamp()
oneDay = 86400
nextUnix = lastUnix + oneDay

for i in forecastSet:
    nextDate = datetime.datetime.fromtimestamp(nextUnix)
    nextUnix += oneDay
    df.loc[nextDate] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()