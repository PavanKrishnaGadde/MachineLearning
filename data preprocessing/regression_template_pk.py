# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 23:32:07 2020

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
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

#fitting the  Regression model 

#predicting the output
Y_pred = regressor.predict(6.5)

#visualizing Regression model ouput
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Predicting the salary using regression')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')


#visualizing output with higer resolution and smoother curve
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, linearRegressor2.predict(polynomial.fit_transform(X_grid)), color = 'green')
plt.title('Predicting the salary using ploynomial regression')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')

