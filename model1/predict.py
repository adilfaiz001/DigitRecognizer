'''
Created on May 10, 2018

@author: adil
'''

'''
Prediction and accuracy calculation for the model. 
'''
import numpy as np
import pandas as pd
import pickle

def predict(parameters,X):
    W,b=parameters
    L=len(W)      
    act=X 
    for b, w in zip(b, W):
        z = np.dot(w, act)+b     
        act=sigmoid(z)
        
    act=softmax(act)
    Y_pred=softmax_reg(act)
    return Y_pred

        
def sigmoid(z):
        return 1/(1+np.exp(-z)) 
 

##softmax returning probability of vector    
def softmax(x):
    x=x.astype(float)
    if x.ndim==1:
        S=np.sum(np.exp(x))
        return np.exp(x)/S
    elif x.ndim==2:
        result=np.zeros_like(x)
        M,N=x.shape
        for n in range(N):
            S=np.sum(np.exp(x[:,n]))
            result[:,n]=np.exp(x[:,n])/S
        return result
    else:
        print("The input array is not 1- or 2-dimensional.")

##returning one for max probability and zero for other 
def softmax_reg(dist):
    dist=dist.astype(float)
    if dist.ndim==1:
        index=np.argmax(dist, axis=0)
        for i in xrange(10):
            if i == index:
                dist[i]=1
            else:
                dist[i]=0
        return dist
    
    elif dist.ndim==2:
        M,N=dist.shape
        for n in xrange(N):
            index=np.argmax(dist[:,n],axis=0)
            for i in xrange(10):
                if i == index:
                    dist[i,n]=1
                else:
                    dist[i,n]=0
        return dist   
        
    

def accuracy(y_true,y_pred):
    delta = y_true-y_pred
    n = np.where(~delta.any(axis=0))[0]
    print n.shape
    t = float(n.shape[0])
    return t/y_true.shape[1]



'''
from load_data import load_data
train_set,valid_set,test_set=load_data()
x,y=test_set
X=x.T
Y=y.T
'''

with open("parameters.pk", 'rb') as fi:
    parameters = pickle.load(fi)


'''    
Y_pred=predict(parameters, X)
print Y_pred
print accuracy(Y,Y_pred)


num = np.argmax(Y_pred[:,1520],axis=0)
num_true= np.argmax(Y[:,1520],axis=0)
print "Predict Number:",num,"\nTure Number:",num_true
'''

df = pd.read_csv('./mnist/test.csv')
df = df.values
X = df.T
print X.shape
y_pred = predict(parameters, X)

lst_pred = np.empty((0,2),int)

for col in xrange(1,y_pred.shape[1]+1):
    
    l = y_pred[:,col-1]
    
    index = np.argmax(l,axis = 0)
    
    lst_pred = np.append(lst_pred,[[col,index]], axis = 0)
    
dataframe = pd.DataFrame(data = lst_pred,columns = ['ImageId','Label'])
dataframe = dataframe.set_index('ImageId')
dataframe.to_csv('test_label.csv')
