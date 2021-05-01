

import numpy as np 
import random 

class Model():
    def __init__(self,learning_rate=1e-1,max_iter=10000,threshold=1e-6,batch_size=100,random_state=0):
        self.learning_rate = learning_rate 
        self.max_iter = max_iter
        self.threshold = threshold 
        self.batch_size = batch_size
        self.random_state = random_state 
        self.layers = []
        self.loss_layer = None
        
    def add(self,layer):
        layer.random_state = self.random_state 
        layer.learning_rate = self.learning_rate 
        self.layers.append(layer)
        
    def forward(self,X,y):
        flow = X
        for layer in self.layers:
            flow = layer.forward(flow)
        loss = self.loss_layer.forward(flow,y)
        return loss
        
    def backward(self):
        v = self.loss_layer.backward(1)
        for layer in  self.layers[::-1]:
            v = layer.backward(v)
            
    def fit(self,X,y,loss,optimizer = "GradientDescent"):
        self.loss_layer = loss
        N = X.shape[0]
        
        losses = []
        if optimizer == "GradientDescent":
            for _ in range(self.max_iter):
                res = self.forward(X,y)
                losses.append(res)
                if res < self.threshold:
                    return 
                self.backward()
                
        elif optimizer == "SGD":
             for _ in range(self.max_iter):
                np.random.seed(self.random_state)
                batch_idx = np.random.randint(0,N,self.batch_size)
                batch_X = X[batch_idx]
                batch_y = y[batch_idx]
                res = self.forward(batch_X,batch_y)
                losses.append(res)
                if res < self.threshold:
                    return 
                self.backward()
        return losses
    
    def predict(self,X):
        for layer in self.layers:
            X = layer.forward(X)
        return X