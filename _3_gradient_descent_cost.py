import numpy as np
import pandas as pd


# how to kinow the best fit line 
'''
by drawing the line which cuts all the points near to it .
and mainly the mean squared error of the lines should be less .
mean squared function also called as cost function .

The lines that are to be done to know which line perfectly fits the points , there is actaully a specific
method to do so . and that is by gradient descent algorithm .



'''


def gradient_descent(x,y):
    m_curr = b_curr = 0 # intials of m and b taken as 0
    iterations = 10000  # how many baby steps u want to take to reach the minima 
    n = len(x)
    
    learning_rate = 0.08 # this is actaully based on trail and error 
    
    for i in range(iterations):
        y_predicted = m_curr * x  + b_curr
        
        cost = (1/n)*sum([val**2 for val in (y-y_pedicted)])
        
        m_derivative = -(2/n)*sum(x*(y-y_perdicted))
        b_derivative = -(2/n)*sum(y-y_perdicted)
        
        m_curr = m_curr - learning_rate * m_derivative
        b_curr = b_curr - learning_rate * b_derivative
        print( "m {} , b {} , iteration {} , cost{}".format(m_curr,b_curr,i,cost) )



x = np.array([1,2,3,4,5,6,7,8])
y = np.array([2,3,4,56,7,78,9])

gradient_descent(x,y)