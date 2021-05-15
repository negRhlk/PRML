"""Model 

    _Model
    Regressor
    Classifier

Todo:
    caluculate hessian 
    baysesian nn 
    implementation of optimzer
"""

import numpy as np 
import random 

from prml.utils.encoder import LabelToOnehot
from prml.utils.util import softmax

class _Model():
    """Model 

    Attributes:
        learning_rate (float) : learning_rate 
        max_iter (int) : number of max iteration when training 
        threshold (float) : threshold 
        batch_size (int) : number of batch size 
        random_state (int) : random state 
        layers (array) : layers  
    """
    def __init__(self,learning_rate=1e-1,max_iter=10000,threshold=1e-6,batch_size=100,random_state=0):
        """Model 

        Args:
            learning_rate (float) : learning_rate 
            max_iter (int) : number of max iteration when training 
            threshold (float) : threshold 
            batch_size (int) : number of batch size 
            random_state (int) : random state 
        """
        self.learning_rate = learning_rate 
        self.max_iter = max_iter
        self.threshold = threshold 
        self.batch_size = batch_size
        self.random_state = random_state 
        self.layers = []
        self.loss_layer = None
        
    def add(self,layer):
        """add 

        Add new layer to the bottom of model 

        Args:
            layer (layer) : layer class which inherits _Layer 
        
        """
        layer.random_state = self.random_state 
        layer.learning_rate = self.learning_rate 
        self.layers.append(layer)
        
    def forward(self,X,y):
        """forward 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (2-D array) : data, shape = (N_samples,N_class or N_target) 
        """
        flow = X
        for layer in self.layers:
            flow = layer.forward(flow)
        loss = self.loss_layer.forward(flow,y)
        return loss
        
    def backward(self):
        """backward  

        back propagation 
        """
        v = self.loss_layer.backward(1)
        for layer in  self.layers[::-1]:
            v = layer.backward(v)
            
    def fit(self,X,y,loss,optimizer = "GradientDescent"):
        """fit 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (2-D array) : data, shape = (N_samples,N_class or N_target)
            loss (loss-layer) : layer which inherits _LossLayer 
            optimizer (str) : "GradientDescent" or "SGC" 
        """
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
        """predict 

        Args:
            X (2-D array) : data shape = (N_samples,N_dims) 
        """
        for layer in self.layers:
            X = layer.forward(X)
        return X

class RegressorNN(_Model):
    pass 

class ClassifierNN(_Model):
    def __init__(self,learning_rate=1e-1,max_iter=10000,threshold=1e-6,batch_size=100,random_state=0):
        """Classifier

        Args:
            learning_rate (float) : learning_rate 
            max_iter (int) : number of max iteration when training 
            threshold (float) : threshold 
            batch_size (int) : number of batch size 
            random_state (int) : random state 
        """
        super(ClassifierNN,self).__init__(learning_rate,max_iter,threshold,batch_size,random_state)
        self.transformer = None 
    
    def _label_to_onehot(self,y): 
        if y.ndim == 1:
            self.transformer = LabelToOnehot() 
            y = self.transformer.fit_transform(y) 
        return y
    
    def _inverse_transform(self,y):
        if self.transformer is None:
            return y
        else:
            return self.transformer.inverse(y) 

    def fit(self,X,y,loss,optimizer = "GradientDescent"):
        """fit 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded.
            loss (loss-layer) : layer which inherits _LossLayer 
            optimizer (str) : "GradientDescent" or "SGD" 
        """
        y = self._label_to_onehot(y) 
        return super(ClassifierNN,self).fit(X,y,loss,optimizer)

    def predict(self,X,return_prob=False):
        """predict 

        Args:
            X (2-D array) : data shape = (N_samples,N_dims) 
            return_prib (bool) : if True, return probabilities 
        
        Returns:
            softmax (2-D array) : classification 
        """
        feature = super(ClassifierNN,self).predict(X)
        prob = softmax(feature)
        if return_prob:
            return prob 
        else:
            onehot = np.zeros_like(prob) 
            clas = prob.argmax(axis = 1)
            for k in range(onehot.shape[1]):
                onehot[clas == k,k] = 1
            return self._inverse_transform(onehot)

