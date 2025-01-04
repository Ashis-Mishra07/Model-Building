import numpy as np
import pandas as pd

'''
Since ML is good in handling the integer data , so how to deal with the string based data.
One way is to convert it into Integer encoding or level Encoding .

If we convert this to integer might be cause issue like 1<2<3 in case of string too occur .

There are two types of categorial data:
    Nominal -> where there is no relation between the category ( male , female)
    Ordinal -> (low medium high)   , (graduate , masters , phd)
'''

# creating dummy variable columns - one hot encoding

#create it 
dummies = pd.get_dummies(df.town) # on which column u want to apply
#now concat it with the main table
merged = pd.concat([df,dummies],axis='columns')
# now drop the main town columns and one column from the newly encoding column (imp) or else it will cause dummy variable trap

final = merged.drop(['town','west windsor'],axis='columns')
