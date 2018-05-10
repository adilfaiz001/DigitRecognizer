'''
Created on May 11, 2018

@author: adil
'''
import numpy as np 
import pickle
from predict import *
import cv2


with open("parameters.pk", 'rb') as fi:
    parameters = pickle.load(fi)
    
def app(X):
    
    Y_pred=predict(parameters, X)
    num = np.argmax(Y_pred, axis=0)
    print "Number - ",np.squeeze(num) 
    

image=cv2.imread("sample.png") 
image= cv2.cvtColor(image, cv2.COLOR_RGB2GRAY )


r= 28.0 / image.shape[1]                    # take care of aspect ration while resizing an image along height or width 
dim = (28,int(image.shape[0] * r))

image = cv2.resize(image,dim,interpolation = cv2.INTER_AREA)

M,N=image.shape
for i in xrange(M):
    for j in xrange(N):
        if image[i,j]<=8:
            image[i,j]=0

m=float(np.max(image))
image=np.divide(image,m)

image=np.reshape(image,[784,1])

app(image)