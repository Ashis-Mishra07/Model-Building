import numpy as np
import pandas as pd

'''

It will split the data into sub parts and the line will be drawn by vector plot in such a way that it will 
put the maximum margin between the points . These distance of the margins are called support vectors 

In 2d Space the boundary is a line 
In 3d Space the boundary is a plane .
In nd Space the boundary is called a hyperplane .

'''

'''
Gamma and Regularization

Gamma 
    i.  High Gamma -> here the curvy line drawn in such a way that it touches the nearer points and collect the margins from it only
                      and excluding the distant points .
                      
    ii. Low Gamma  -> here the line will cut the points and it will also consider the distant points too .(more accuracy and time computational)
    
Regularization ( C )
    i. High Regularization -> here the curvy line will divide each parameter into separates that it avoids any classification error (no error and curvy line)
    ii. Low Regularization -> here the straight line is drawn but it may take some errors .(some error and smoother line )
    

'''


'''

 if on x axis or y axis is not possible , then we can draw by z axis
  z = x^2 + y^2 
   
  z transformation is called kernel
  
'''

# learning from iris dataset
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris() # name changing

df = pd.DataFrame(iris.data , columns=iris.feature_names ) # show the sepal and petal length and width

df['target'] = iris.target # this will append the target column

df['flower_name'] = df.target.apply(lambda x:iris.target_names[x])


# now its time to draw the graphs

df0 = df[df.target ==0 ]
df1 = df[df.target ==1 ]
df2 = df[df.target ==2 ]






