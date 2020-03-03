# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 00:25:28 2020

@author: pavankrg
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#importing dataset

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Thomson Sampling

N = 10000
d = 10
no_of_reward_1 = [0] * d
no_of_reward_0 = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    max_random = 0
    ad = 0
    for i in range(0,d):
        random_beta = random.betavariate(no_of_reward_1[i] +1,no_of_reward_0[i]+1 )
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if(reward == 1):
        no_of_reward_1[ad] = no_of_reward_1[ad] +1 
    else:
        no_of_reward_0[ad] = no_of_reward_0[ad] +1 

    total_reward = total_reward + reward
    
    
#Visualizing results
    
plt.hist(ads_selected)
plt.title('histogram of ad selected')
plt.xlabel('ad')
plt.ylabel('no of times ad is selected')