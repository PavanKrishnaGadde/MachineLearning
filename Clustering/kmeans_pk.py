# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:19:10 2020

@author: pavankrg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# elbow method to find optimum number of clusters

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')


#Applying K-means
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter = 300, n_init=10, random_state=0)
Y_pred = kmeans.fit_predict(X)

#data visualization

plt.scatter(X[Y_pred == 0,0], X[Y_pred == 0, 1], s = 100, c = 'red', label ='cluster1')
plt.scatter(X[Y_pred == 1,0], X[Y_pred == 1, 1], s = 100, c = 'cyan', label ='cluster2')
plt.scatter(X[Y_pred == 2,0], X[Y_pred == 2, 1], s = 100, c = 'magenta', label ='cluster3')
plt.scatter(X[Y_pred == 3,0], X[Y_pred == 3, 1], s = 100, c = 'yellow', label ='cluster4')
plt.scatter(X[Y_pred == 4,0], X[Y_pred == 4, 1], s = 100, c = 'black', label ='cluster5')

