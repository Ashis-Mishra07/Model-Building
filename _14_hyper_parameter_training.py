'''
Methods Learnt -> Random Forest , Logisticx Regression , Decision Tree , Naive Bayes , SVM

suppose i am choosing svm , and in it choosing optimal value of kernel is called hyperparameter tuning .

 But we dont know which method to apply and what value for c is to be choosen , 
 so can be made by two for loops , one loop for model finding , and second for optimal C value
 
 
But the above method will run out of time because two loops time will be more , instead we can use GridSerachCV

'''

from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},cv = 5 , return_train_score = False) # how many cross validation 

clf.fit(iris.data , iris.target)
clf.cv_results_


# to make it a dataframe
df = pd.DataFrame( clf.cv_results_ )

# properties
clf.best_score_
clf.best_params_


# if the data will be high then number of outputs or possibilities will be high , but we want randomly do few
# use RandomisedSearchCV

from sklearn.model_selection import RandomisedSearchCV
rs = RandomisedSearchCV(svm.SVC(gamma='auto'),{
    'C':[1,10,20],
    'kernel':['rbf','linear']
},
    cv = 5 , 
    return_train_score = False,
    n_iter = 2 # this means random 2 
)

rs.fit(iris.data , iris.target)
df = pd.DataFrame( clf.cv_results_ )[[ 'param_c','param_kernel','mean_test_score' ]]










# now how to choose best model for a problem
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params={
    'svm':{
        'model':svm.SVC(gamma='auto'),
        'params':{
            'C':[1,10,20],
            'kernel':['rbf','linear']
        }
    },
    'Random_forst':{
        'model':RandomForestClassifier(),
        'params':{
            'n_estimators':[1,10,20]
        }
    },
    'svm':{
        'model':LogisticRegression(solver='liblinear',multi_class ='auto'),
        'params':{
            'C':[1,10,20]
        }
    },
}

scores =[]
for model_name , mp in model_params.items():
    clf = GridSerachCV(mp['model'],mp['params'],cv =5,return_train_score=False)
    clf.fit(iris.data,iris.target)
    scores.append({
        'model':model_name,
        'best_score':clf.best_score_,
        'best_params':clf.best_params_
    })
    
    
# making dataframe to know
df = pd.DataFrame(scores , columns=['model','best_score','best_params'])
