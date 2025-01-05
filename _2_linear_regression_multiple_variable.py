import numpy as np
import pandas as pd

from sklearn import linear_model

price = m1*area + m2*bedrooms + m3*age + b

# How to handle the NaN valuea and Linear Regre using multiple variables


# filling the bedroom section

import math
median_bedrooms = math.floor(df.bedrooms.median())

df.bedrooms = df.bedrooms.fillna(median_bedrooms)

reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)

# to know about the coefficients m1 m2 m3 and intercept b
reg.coef_
reg.intercept_

# now how to predict
reg.predict([[3000,3,40]])
