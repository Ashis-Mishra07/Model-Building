import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

# Load the data
df = pd.read_csv('data.csv')

plt.scatter(df.x, df.y, color='red', marker='+')
plt.xlabel('area')
plt.ylabel('price')
plt.title('Data')

# By observibg the model that it may follow linear regression model
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price) # means training the data

# here it trained so now use for prediction
reg.predict(3300)

# to show the slope and intercept  of the line of linear regression model so used
reg.coef_ # slope
reg.intercept_ # intercept
# and price = m*area + b

# to predict the price of the area from an array
area_df = pd.read_csv('areas.csv')
p = reg.predict(area_df)
area_df['price'] = p
area_df.to_csv('prediction.csv', index=False)




# to plot the line of linear regression model
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.xlabel('area')
plt.ylabel('price')
plt.title('Data')

