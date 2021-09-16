# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 15:43:10 2021

@author: hp
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 
from sklearn.datasets import make_regression
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
#%%
#%%
X, y = make_regression(n_samples= 100 ,n_informative=3, n_features=5)
#%%
print(X.shape , y.shape)
#%%
df = pd.DataFrame(
{'Feature_1':X[:,0],
'Feature_2':X[:,1],
'Feature_3':X[:,2],
'Feature_4':X[:,3],
'Feature_5':X[:,4],
'Target':y
}
)
df.head()
#%%
print(df.describe())

#%%
pca = PCA(n_components=3)
pca.fit(X)

#%%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#%%
#scale the training and testing data
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:1]

#%%
#train PCR model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:1], y_train)

#%%
#calculate RMSE
pred = regr.predict(X_reduced_test)
print(np.sqrt(mean_squared_error(y_test, pred)))

#%%
print(r2_score(y_test,pred))

#%%











