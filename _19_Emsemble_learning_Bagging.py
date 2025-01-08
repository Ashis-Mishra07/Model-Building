'''

From a whole big sample set of dataset take out few samples of it (like 70% of the whole)
Now take out such samples for such n times  and calculate the losgistoic Regression of each 
and then find the majority of the all testing samples . 


In Bootstrap Aggregation the average value that been taken from all the samples is done .
Bagging is called Bootstrap Aggregation .

Random Forest is one of the Bagging technique with one distinction is that data as well as features are sampled 
means row as well as columns are sampled .

In Random Forest technique , multiple Decision trees are made from the main one by taking random columns 
from the main table and then sampled it and find the average .


Bagging -> Indivisual model can be anything (SVM , Knn , Logistic Regression , etc)
Bagged Trees -> Each model is a tree (Random Forest )


'''

import pandas as pd

df = pd.read_csv('diabetes.csv')

X = df.drop('Outcome',axis='columns')
y = df.Outcome


# might be not scaled upto the mark

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)



from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test  = train_test_split(X_scaled , y , test_size = 0.2 , random_state =10 , stratify=y  )


# we are using decison tree classification model 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

scores = cross_val_score( DecisionTreeClassifier() , X , y , cv=5 )
scores.mean()    # this is stand along model 





from sklearn.ensemble import BaggingClassification 

bag_model = BaggingClassification(
    base_estimator = DecisionTreeClassifier() ,  # doing this classifiers 
    n_estimators = 100 ,   # taking out the samples 100 times 
    max_samples =0.8 ,     # finding out 80% of the whole samples 
    oob_score = True ,     
    random_state =0
)
'''
This oob score is out of bag , actually while diving the samples it takes few sample in cache for testing .
and this sample is not used while training , later while testing it is used ,
'''
bag_model.fit(X_train , y_train)
bag_model.oob_score_   # we don't even write X_test , y_test  , this is bagging model 



# we can also do it by cross  validation
bag_model = BaggingClassification(
    base_estimator = DecisionTreeClassifier() ,  # doing this classifiers 
    n_estimators = 100 ,   # taking out the samples 100 times 
    max_samples =0.8 ,     # finding out 80% of the whole samples 
    oob_score = True ,     
    random_state =0
)

scores = cross_val_score( bag_model , X , y , cv=5 )
scores.mean()    




# Can be done by Random Forest , this internally uses the Bagging technique
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 100 , random_state = 0 )
scores = cross_val_score( rf_model , X , y , cv=5 )
scores.mean()    





























