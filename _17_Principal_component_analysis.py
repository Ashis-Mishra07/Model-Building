'''
Technique used to reduce dimension . 
It is used to reduce the number of features in the dataset.

What if we get rid of non important features ?
    Faster training and inference .
    Data visualization becomes easier .

The basic meaning is that when we try to visualise something for example any digits series number dataset .
Some part of the pixels are basically unuse because they are not contributing to the image . so why 
to use them rather reducing it is a good option so , here reducing means reducing the dimensions .

PCA : Pricipal Component Analysis
Is a process of figuring out most important features or principal components that has 
the most impact on the target variable . It creates a new component removing the less important
features and keeping the most important features and working on it .

Drawing the PCA :
i. Draw a horizontal line cuttig the maximum points where the variance is maximum .
ii. Draw a line perpendicular to the first line and cut the maximum points where the variance is maximum .

 
For Example:

Suppose we have a dataset with numbers and it is having pixel1 pixel2 pixel3 .... pixel64 and Output for each number
Now it is observed that the some pixel number are always showing 0 value so removing it .

PCA(n_components = x ) This PCA value will extract the x most important features from the dataset 
                       which is having the most impact on the target variable .
                       It will result in formation of PC1 PC2 PC3 .... PCx . shows the most important features .


                       
NOTE:
If the scale is not upto the mark like x axis is with millions of data and y axis is having only few .
so graph will show very skewed type . So it is important to scale the data before applying PCA or
Accuracy might drop .

PCA is used to reduce the dimensions of the dataset and it is used to reduce the dimensions of the dataset.

'''

import pandas as pd
from sklearn.datasets import load_digits

dataset = load_digits()
dataset.keys()   # shows u the keys the dataset is having .

dataset.data[0] # shows the 1st data of the dataset and it will be in 1D format .
dataset.data.reshape(8,8) # it will reshape it and shwo it in form of a matrix .


from matplotlib import pyplot as plt

plt.gray()
plt.matshow(dataset.data.reshape(8,8)) # it will show u in gray format 

df = pd.DataFrame(dataset.data,colums=dataset.feature_names )
# the above will show the dataset separating into pixel1 pixel2 pixel3 .... pixel64

X = df
y = dataset.target # from 0 to 9

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()  # to make the scale uniform between 0 to 1 value .
X_scaled = scalar.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split( X_scaled , y , test_size=0.2 , random_state=42 )

# By Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)







# by PCA 
from sklearn.decomposition import PCA
# pca = PCA(n_components=10) # it will take 10 most important features from the dataset .
pca = PCA(0.95) # it will take 95% of the data from the dataset .

X_pca = pca.fit_transform(X) # by the time it will remove the  less important features 
# It is not that it will randomly choose the columns , it will calculate and then choose the best fit of it .

# to know about the variance precentage of the data
pca.explained_variance_ratio_
pca.n_components_ # it will show the number of components choosen by the PCA , which is similar to X_pca.shape

X_train,X_test,y_train,y_test  = train_test_split(X_pca , y , test_size=0.2,random_state=42)

# By Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() # if it limits exceeds then add  .....  max_iter = 1000 
model.fit(X_train,y_train)
model.score(X_test,y_test)






















