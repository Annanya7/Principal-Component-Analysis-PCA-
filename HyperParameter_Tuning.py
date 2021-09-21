# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:59:43 2021

@author: hp
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV


#%%
X,y = make_regression(n_samples = 1000, n_features = 1)
X2,y2 = make_regression(n_samples = 1000, n_features = 1, noise = 100.0)

plt.scatter(X,y,color='red')
#print(X)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#%%

plt.hist(np.squeeze(X))
plt.show()
plt.hist(np.squeeze(X2))

#%%

X_train , X_test , y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=30)

#%%

tuned_parameters = [{'fit_intercept': ['True'],'normalize': ['True']},{'fit_intercept': ['False'],'normalize': ['True']}]

#%%

model= GridSearchCV(LinearRegression(), tuned_parameters, scoring='r2')

#%%

model.fit(X_train, y_train)

#%%
print(model.best_params_)