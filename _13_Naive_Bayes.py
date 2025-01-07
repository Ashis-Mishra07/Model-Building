# Naive Bayes is used in Email Spam , Face Detection , Weather Detection , New Article Categorisation

import pandas as pd
df = pd.read_csv('titanic')

target = df.survived
inputs = df.drop['survived',imnplace=True]

# for sex columns making it by one hot encoding
dummy = pd.get_dummies(inputs.sex)
inputs = pd.concat([inputs , dummy] , axis ='column')
inputs.drop('sex',axis='column',inplace=True)

# checking if any na values
inputs.column[inputs.isna().any()] # this will return the column name which is having NaN values i.e. age here

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

#train and test
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(inputs,target , test_size=0.2)


# now using naive bayes classes
from sklearn.naive_bayes import GaussanNB
model = GaussanNB()

model.fit(X_train , y_train)
model.score(x_test,y_test)

model.predict(X_test[:10]) # here it give an array and used to check with y_test to find the prediction





















