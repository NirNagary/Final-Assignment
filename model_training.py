
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 17:12:03 2023

@author: annak
"""
import re
import csv
import pandas as pd
import numpy as np 
from datetime import timedelta
from datetime import datetime
import datetime as dt
import unicodedata
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import ElasticNet
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer
import pickle



####  splitting the data ## 
X = data.drop('price',axis = 1)
X = X.drop(['description ','num_of_images','handicapFriendly ','hasBars ','furniture ','entrance_date'],axis =1)
#X = X.drop(['description ','city_area'],axis =1)
y = data.price






cat_cols = ['City','type','city_area','Street','condition ']  # List of categorical column names
#num_cols = ['room_number','Area','num_of_images','hasElevator ','hasParking ','hasBars ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','handicapFriendly ','floor','total_floors']  # List of numerical column names
#cat_cols = ['City','type','Street','condition ','furniture ','entrance_date']  # List of categorical column names
num_cols = ['room_number','Area','hasElevator ','hasParking ','hasStorage ','hasAirCondition ','hasBalcony ','hasMamad ','floor','total_floors']  # List of numerical column names



# Feature Engineering
log_transformer = FunctionTransformer(np.log1p, validate=True)
interaction_transformer = PolynomialFeatures(degree=2, interaction_only=True)

numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('log_transform', log_transformer),
    ('scaling', RobustScaler())
])

poly_pipeline = Pipeline([
    ('polynomial_features', interaction_transformer),
    ('scaling', RobustScaler())
])

categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='constant', add_indicator=False, fill_value='missing')),
    ('one_hot_encoding', OneHotEncoder( handle_unknown='ignore'))
])





column_transformer = ColumnTransformer([
    ('numerical_pipeline', numerical_pipeline, num_cols),
    ('categorical_preprocessing', categorical_pipeline, cat_cols),
], remainder='drop')


pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('model', ElasticNet(alpha=0.01, l1_ratio=0.9))
])

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)


# Evaluate the pipeline using cross-validation
scores = cross_val_score(pipe_preprocessing_model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

pipe_preprocessing_model.fit(X,y)

rmse_scores = np.sqrt(-scores) 

mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)

print('Mean RMSE: %.3f' % mean_rmse)
print('Standard Deviation of RMSE: %.3f' % std_rmse)

pickle.dump(pipe_preprocessing_model, open("trained_model.pkl","wb"))











