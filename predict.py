'''
Created on May 10, 2018

@author: adil
'''
import numpy as np
import pickle

def predict(parameters,X):
    W,b=parameters
    L=len(W)      
    act=X 
    for b, w in zip(b, W):
        z = np.dot(w, act)+b
        act=sigmoid(z)
        
    act=softmax(act)
    Y_pred=pred(act)
    return Y_pred
        
def sigmoid(z):
        return 1/(1+np.exp(-z)) 
 
    
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


def pred(dist):
    dist=dist.astype(float)
    if x.ndim==1:
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


from load_data import load_data
train_set,valid_set,test_set=load_data()
x,y=valid_set
X=x.T
Y=y.T
with open("parameters.pk", 'rb') as fi:
    parameters = pickle.load(fi)
    
Y_pred=predict(parameters, X)
print X.shape
print accuracy(Y,Y_pred)
