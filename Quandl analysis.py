# -*- coding: utf-8 -*-

import pandas as pd
import Quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = Quandl.get('WIKI/GOOGL')
df               = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume', ]]
df['HL_PCT']     = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df               = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.1 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

print df.shape

X = np.array(df.drop(['label'], 1))
print X.shape

y = np.array(df['label'])
print y.shape
#
X = preprocessing.scale(X)
#X = X[:-forecast_out]
#X_lately = X[-forecast_out:]
df.dropna(inplace = True)
print df.shape
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
print clf.score(X_train, y_train)
print clf.score(X_test, y_test)

# forecast_set = clf.predict(X_lately)
# print forecast_set, forecast_out