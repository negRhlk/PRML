"""Layers 

    Dense

    Activation Layers:
        Tanh
        Relu
    
    Loss Layers:
        MeanSquaerdErroor 
        SigmoidCrossEntropy 
        SoftmaxCrossEntropy 
"""

import numpy as np 
from abc import ABCMeta,abstractclassmethod

from prml.utils.util import sigmoid,softmax,binary_cross_entropy,cross_entropy


class _Layer(metaclass=ABCMeta):
    """_Layer

    abstract class for layers 
    """
    def __init__(self): 
        self.initialize = False 
        self.learning_rate = None 
        self.random_state = None  
    
    @abstractclassmethod
    def forward(self,X):
        pass 

    @abstractclassmethod
    def backward(self,loss):
        pass 

class Dense(_Layer):
    """Dense layer 

    Fully Conncted layer or Affine Layer 

        y = Wx + b 
    
    Attributs:
        output_size (int) : output data shape is (N_samples,output_shape) 
        W,b (array) : parameter for this layer 
        X (2-D array) : data which is used the last time data was forward
    """
    def __init__(self,output_size):
        super(Dense,self).__init__() 
        self.output_size = output_size
        self.X = None
        self.W = None 
        self.b = None
        
    def forward(self,X):
        """forward 

        Args:
            X (2-D array) : data, shape is (N_samples,N_dims)
        
        Returns:
            X (2-D array) : data, shape is (N_samples,output_size)
        """
        self.X = X
        if not self.initialize:
            np.random.seed(self.random_state)
            self.W = np.random.randn(X.shape[1],self.output_size)
            self.b = np.random.randn(self.output_size)
            self.initialize = True 
        return np.dot(X,self.W) + self.b 
    
    def backward(self,loss):
        """backward 

        Args:
            loss (2-D array) : loss, shape is  (N_samples,output_shape), N_samples equals X.shape[0] at the last time forward() was called. 
        
        Returns:
            dX (2-D array) : the last layer's loss, shape is (N_samples,N_dims) 
        """
        dX = np.dot(loss,self.W.T)
        dW = np.dot(self.X.T,loss)
        db = loss.sum(axis = 0)
        self.W -= self.learning_rate*dW
        self.b -= self.learning_rate*db
        return dX

class _ActivationLayer(metaclass=ABCMeta):
    def __init__(self):
        pass 
    @abstractclassmethod
    def forward(self,X):
        pass 
    @abstractclassmethod
    def backward(self,loss):
        pass 

class Tanh(_ActivationLayer):
    """Tanh Layer

    
    """
    def __init__(self):
        super(Tanh,self).__init__()
        self.X = None
    def forward(self,X):
        """forward

        Args:
            X (2-D array) : data,shape is (N_samples,N_dims) 
        
        Returns:
            relu (2-D array) : Tanh(X)
        """
        self.X = X
        plus = X > 0 
        minus = X <= 0 
        Y = np.zeros_like(X) 
        Y[plus] = (1 - np.exp(-2*X[plus]))/(1 + np.exp(-2*X[plus]))
        Y[minus] = (np.exp(2*X[minus]) - 1)/(np.exp(2*X[minus]) + 1) 
        return Y
    def backward(self,loss):
        """backward 

        Args:
            loss (2-D array) : loss, shape is  (N_samples,input_shape), N_samples equals X.shape[0] at the last time forward() was called. 
        
        Returns:
            dX (2-D array) : the last layer's loss, shape is (N_samples,input_shape) 
        """
        plus = self.X > 0 
        minus = self.X <= 0 
        tmp = np.zeros_like(self.X) 
        tmp[plus] = 2*np.exp(-self.X[plus])/(1 + np.exp(-2*self.X[plus]))
        tmp[minus] = 2*np.exp(self.X[minus])/(1 + np.exp(2*self.X[minus])) 
        dX = loss*np.square(tmp) 
        return dX
    
class Relu(_ActivationLayer):
    """Relu Layer 

    Relu is non-differentiable at x = 0. Derivative value at x = 0 is define as 1 in this layer. 
    """
    def __init__(self):
        super(Relu,self).__init__() 
        self.X = None 

    def forward(self,X):
        """forward

        Args:
            X (2-D array) : data,shape is (N_samples,N_dims) 
        
        Returns:
            relu (2-D array) : Relu(X)
        """
        self.X = X
        return np.where(X > 0,X,0)
    def backward(self,loss):
        """backward 

        Args:
            loss (2-D array) : loss, shape is  (N_samples,input_shape), N_samples equals X.shape[0] at the last time forward() was called. 
        
        Returns:
            dX (2-D array) : the last layer's loss, shape is (N_samples,input_shape) 
        """
        dX = np.zeros_like(self.X)
        dX[self.X >= 0] = 1
        return loss*dX

class _LossLayer(metaclass=ABCMeta):
    def __init__(self):
        pass 
    @abstractclassmethod
    def forward(self,predict,target):
        pass 
    @abstractclassmethod
    def backward(self,loss):
        pass 

class MeanSquaredError(_LossLayer):
    """MeanSquaredError

    loss layer  used in regression problems 
    """
    def __init__(self):
        super(MeanSquaredError,self).__init__()
        self.predict = None 
        self.target = None
    def forward(self,predict,target):
        """forward 

        Args:
            predict (array) : predict values 
            target (array) : ground truth 
        
        Return:
            loss (float) : mean squared loss 
        """
        self.predict = predict
        self.target = target  
        return np.mean((predict - target)**2)/2
    def backward(self,loss):
        """backward 

        Args:
            loss (2-D array) : 1 
        Returns:
            dX (2-D array) : the last layer's loss, shape is (N_samples,input_shape) 
        Note:
            loss layer is always receive 1 when backward  
        """
        shape = self.target.shape[0]
        dX = loss*(self.predict - self.target)/shape
        return dX

class SigmoidCrossEntropy(_LossLayer): 
    """sigmoid activation function and cross entropy loss 

    this layer is sigmoid layer which has binary cross entropy loss 
    """
    def __init__(self): 
        super(SigmoidCrossEntropy,self).__init__() 
        self.predict = None 
        self.target = None 
    
    def forward(self,X,target): 
        """forward

        Args:
            X (2-D array) : data,shape is (N_samples,2)
            target (2-D array) : data, shape is (N_samples,2) 
        
        Returns:
            loss (float) : cross entropy of sigmoid(X) and target 
        """
        self.predict = softmax(X) 
        self.target = target 
        return binary_cross_entropy(target,self.predict) 
    
    def backward(self,loss): 
        """backward 

        Args:
            loss (2-D array) : 1 
        Returns:
            dX (2-D array) : the last layer's loss, shape is (N_samples,input_shape) 
        Note:
            loss layer is always receive 1 when backward  
        """
        shape = self.target.shape[0] 
        return loss*(self.predict - self.target)/shape 

class SoftmaxCrossEntropy(_LossLayer): 
    """Softmax Cross Entropy 

    this layer is softmax layer which has cross entropy loss 
    """

    def __init__(self): 
        super(SoftmaxCrossEntropy,self).__init__()
        self.predict = None 
        self.target = None 
    
    def forward(self,X,target): 
        """forward 

        Args:
            X (2-D array) : data, shape = (N_samples,N_class)
            target (2-D array) : ground truth, shape = (N_samples,N_class)
        Return:
            loss (float) : cross entropy loss 
        """
        self.predict = softmax(X)  
        self.target = target 
        return cross_entropy(target,self.predict) 
    
    def backward(self,loss):
        """backward 

        Args:
            loss (2-D array) : 1 
        Returns:
            dX (2-D array) : the last layer's loss, shape is (N_samples,input_shape) 
        Note:
            loss layer is always receive 1 when backward  
        """
        shape = self.target.shape[0]
        return loss*(self.predict - self.target)/shape 