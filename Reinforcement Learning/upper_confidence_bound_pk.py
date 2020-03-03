# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:17:02 2020

@author: pavankrg
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#importing dataset

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#UCB
N = 10000
d = 10
no_of_selections = [0] * d
sums_of_rewards = [0] * d
ads_selected = []
total_reward = 0

for n in range(0, N):
    max_upper_bound = 0
    ucb_ad = 0
    for i in range(0,d):
        if (no_of_selections[i] > 0):
            average_reward_i = sums_of_rewards[i]/ no_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/ no_of_selections[i])
            upper_confidence_bound = average_reward_i + delta_i
        else:
            upper_confidence_bound = 1e400
        if upper_confidence_bound > max_upper_bound:
            max_upper_bound = upper_confidence_bound
            ucb_ad = i
    ads_selected.append(ucb_ad)
    no_of_selections[ucb_ad] = no_of_selections[ucb_ad] + 1
    reward = dataset.values[n,ucb_ad]
    sums_of_rewards[ucb_ad] = sums_of_rewards[ucb_ad] + reward
    total_reward = total_reward + reward
    
    
#Visualizing results
    
plt.hist(ads_selected)
plt.title('histogram of ad selected')
plt.xlabel('ad')
plt.ylabel('no of times ad is selected')

    