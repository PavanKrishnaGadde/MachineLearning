# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 21:44:32 2020

@author: pavankrg
"""

#importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting the data into train and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=0)

#Fitting simple linear regression modal to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Predicting the test set results
Y_pred = regressor.predict(X_test)

#visualising train set results
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()

#visualising test set results
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color= 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()