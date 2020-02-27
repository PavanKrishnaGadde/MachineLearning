# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 00:19:12 2020

@author: pavankrg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#Finding optimal number of clusters

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendrogram')
plt.xlabel('Customers')
plt.ylabel('Ecludiean distances')

# --> ward method minizes the variance within each cluster

from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering (n_clusters=5, affinity='euclidean', linkage='ward')

Y_pred = hc.fit_predict(X)

#Data Visualization

plt.scatter(X[Y_pred == 0,0], X[Y_pred == 0, 1], s = 100, c = 'red', label ='cluster1')
plt.scatter(X[Y_pred == 1,0], X[Y_pred == 1, 1], s = 100, c = 'cyan', label ='cluster2')
plt.scatter(X[Y_pred == 2,0], X[Y_pred == 2, 1], s = 100, c = 'magenta', label ='cluster3')
plt.scatter(X[Y_pred == 3,0], X[Y_pred == 3, 1], s = 100, c = 'yellow', label ='cluster4')
plt.scatter(X[Y_pred == 4,0], X[Y_pred == 4, 1], s = 100, c = 'black', label ='cluster5')

