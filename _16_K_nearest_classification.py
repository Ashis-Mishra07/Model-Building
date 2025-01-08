'''
Working of K_nearest_classification

i. Form the cluster based on the parameters and it will form the points in the graph
ii. Now decide the value of K , then take a random point and find the distance between the 
    scatter points and the random point .

iii. Now as per the K value take all the distances and sort them in ascending order
iv.  Now the point to which the maximum number of points are closer , that point will belongs to that grp .

'''


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

iris.feature_names # it will show u the feature names like petal and sepal width and length

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target']=iris.target

# do the scatter plots for the types of the data
plt.scatter(df['sepal width (cm)'], df['sepal length (cm)'], color='green',marker='+')
plt.scatter(df['petal width (cm)'], df['petal length (cm)'], color='red',marker='*')

from sklearn.model_selection import train_test_split

X = df.drop(['target','flower_name'] , axis=1) 
y = df.target

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=1)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3) # this will create the 3 clusters means k = 3
knn.fit(X_train, y_train)
knn.score(X_test, y_test) # 0.9666

# choose the optimal value of K

#  to see the accuracy of the model for different values of K we will use confusion matrix
from sklearn.metrics import confusion_matrix

y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))  # shows the report of the prediction




