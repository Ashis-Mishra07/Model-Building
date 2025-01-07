
'''
L1 and L2 regularization are used to overcome the overfitting issue .

suppose drawing a scatter plot age vs matches own

for underfit -> linear decreasing line  .....  match_won = theta0 + theta1*age
for overfit  -> zig zag line  .....  match_won = theta0 + theta1*age + theta2*age^2 + theta3*age^3 + theta4*age^4 
for balancedfit -> curve line ..... match_won = theta0 + theta1*age + theta2*age^2

when theta3 qand theta4 are very low means tends to zero then the line becomes curvy from zig zag

The main differnece between the L1 and L2 regularization is that
the mean square error gets added with lambda * sum(theta^2) for L2 regu
the mean square error gets added with lambda * sum(|theta|) for L1 regu

 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('./housing.csv')
dataset.isna().sum() # it will check the rows having null values 
# for few rows fill it with zero , few with mean , few drop them 
cols_to_fill_zero =[ 'total_bedrooms','Car','Distance']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

# for text columns , we need to convert them to numbers by dummy encoding
dataset = pd.get_dummies(dataset, drop_first=True)

X = dataset.drop('Price', axis=1)
y = dataset['Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_test, y_test) # 0.14
reg.score(X_train, y_train) # 0.67  # this is overfit because of large difference in percentage

# Lasso regression is L1 regularization
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(X_train, y_train)
lasso_reg.score(X_test, y_test) # 0.66
lasso_reg.score(X_train, y_train) # 0.67 # this is balanced fit because of less difference in percentage

# Ridge regression is L2 regularization
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, y_train)
ridge_reg.score(X_test, y_test) # 0.66 # almost same as Lasso
















