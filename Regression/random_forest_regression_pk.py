# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:39:25 2020

@author: pavankrg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,2].values

#fitting the  Random Forest Regression model 

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=100,random_state=10 )
regressor.fit(X,Y)

#predicting the output
Y_pred = regressor.predict(np.array([[7.6]]))




#visualizing output with higer resolution and smoother curve
X_grid = np.arange(min(X), max(X),0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Predicting the salary using Random Forest regression')
plt.xlabel('Xlabel')
plt.ylabel('Ylabel')

