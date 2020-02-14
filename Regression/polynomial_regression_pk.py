# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 22:20:36 2020

@author: pavankrg
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,2].values

#Linear Regression
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X, Y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree = 4)
X_ploy  = polynomial.fit_transform(X)

linearRegressor2 = LinearRegression()
linearRegressor2.fit(X_ploy, Y)

#visualizing linear Regression
plt.scatter(X, Y, color = 'red')
plt.plot(X, linearRegressor.predict(X), color = 'green')
plt.title('Predicting the salary using simple linear regression')
plt.xlabel('Position')
plt.ylabel('Salary')

#visualizing ploynomial Regression
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color = 'red')
plt.plot(X_grid, linearRegressor2.predict(polynomial.fit_transform(X_grid)), color = 'green')
plt.title('Predicting the salary using ploynomial regression')
plt.xlabel('Position')
plt.ylabel('Salary')


#prediciting a salary using linear regression
linearRegressor.predict([6.5])

#predicting a salary using ploynomial regression model
linearRegressor2.predict(polynomial.fit_transform(6.5))