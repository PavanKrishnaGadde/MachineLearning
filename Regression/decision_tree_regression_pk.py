# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:20:13 2020

@author: pavankrg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,2].values

#fitting the  Regression model 

from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,Y)

#predicting the output
Y_pred = regressor.predict(np.array([[7.6]]))

#visualizing Regression model ouput
plt.scatter(X, Y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'green')
plt.title('Predicting the salary using Decision Tree regression')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')


#visualizing output with higer resolution and smoother curve
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Predicting the salary using ploynomial regression')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')
