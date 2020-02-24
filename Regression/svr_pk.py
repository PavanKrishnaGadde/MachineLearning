# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:34:34 2020

@author: pavankrg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,2].values

#Splitting the data into train and test sets
#from sklearn.model_selection import train_test_split
#X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X)
y_train = scaler_y.fit_transform(Y.reshape(-1,1))


#fitting the  SVR model  to dataset

from sklearn.svm import SVR

regressor = SVR(kernel = 'rbf')

regressor.fit(X_train, y_train)


#predicting the output
Y_pred = scaler_y.inverse_transform(regressor.predict(scaler_X.transform(np.array([[6.5]]))))

#visualizing Regression model ouput
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Predicting the salary using SVR')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')

X_grid = np.arange(min(X_train), max(X_train),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Predicting the salary using ploynomial regression')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')



