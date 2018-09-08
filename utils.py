import numpy as np
import pandas as pd
def rectified_linear_unit(x):
    if x<0:
        return(0)
    else :
        return(x)

def grad_rectified_linear_unit(x):
    x[x>0] = 1
    x[x<=0] = 0
    return x
    
def one_hot(label):
    arr = np.zeros(shape = (10,1))
    arr[label] = 1
    return arr

def categorical_cross_entropy(y_hat, label):
    return -sum(np.multiply(np.log(y_hat), one_hot(label)))

def stable_softmax(X):
    temp2 = np.exp(X - np.max(X))
    return temp2 / np.sum(temp2)

def decay_alpha(i):
    
    return( 0.01 if i <=12 else (0.001 if i <= 24 else (0.0001)) )
    
def accuracy(y_hat, true):
    return 100*sum(np.where(y_hat == np.reshape(true,len(true)), 1,0))/len(y_hat)