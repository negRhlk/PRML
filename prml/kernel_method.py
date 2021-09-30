"""kernel methods 

chapter6 
BaseKernelMachine
DualRegression 
NadarayaWatson 

"""

import numpy as np 
from prml.utils.util import sigmoid,kappa
from prml.linear_classifier import Classifier
from prml.kernel_func import BaseKernel,LinearKernel,GaussianKernel,SigmoidKernel,RBFKernel,ExponentialKernel,GramMatrix

class BaseKernelMachine():
    """BaseKernelMachine 

    Attributes:
        kernel_func (function) : kernel function k(x,y) 
        gram_func (function) : function which make gram matrix 

    """

    def __init__(self,kernel="Linear",sigma=1.0,a=1.0,b=0.0,h=None,theta=1.0):
        """

        Args:
            kernel (string) : kernel type (default "Linear"). you can choose "Linear","Gaussian","Sigmoid","RBF","Exponential"
            sigma (float) : for "Gaussian" kernel 
            a,b (float) : for "Sigmoid" kernel
            h (function) : for "RBF" kernel 
            theta (float) : for "Exponential" kernel

        """
        self.kernel_func = None
        if kernel == "Linear":
            self.kernel_func = LinearKernel() 
        elif kernel == "Gaussian":
            self.kernel_func = GaussianKernel(sigma=sigma)
        elif kernel == "Sigmoid":
            self.kernel_func = SigmoidKernel(a=a,b=b) 
        elif kernel == "RBF":
            if h is None:
                raise ValueError("if kernel is 'RBF', h must not be None.")
            self.kernel_func = RBFKernel(h=h)
        elif kernel == "Exponential":
            self.kernel_func = ExponentialKernel(theta=theta)
        else:
            raise ValueError(f"kernel '{kernel}' is inappropriate")
        self.gram_func = GramMatrix(kernel,sigma,a,b,h,theta)

class DualRegression(BaseKernelMachine):
    """DualRegression 

    Attributes:
        kernel_func (function) : kernel function k(x,y) 
        gram_func (function) : function which make gram matrix 
        lamda (float) : regularization parameter
        dual_weight (2-D array) : weight 
        X (2-D array) : explanatory variable,shape = (N_samples,N_dim)

    """
    def __init__(self,lamda=0.1,kernel="Linear",sigma=0.1,a=1.0,b=0.0,h=None,theta=1.0):
        """

        Args:
            lamda (float) : regularization parameter, (we cannot use lambda)
            kernel (string) : kernel type (default "Linear"). you can choose "Linear","Gaussian","Sigmoid","RBF","Exponential"
            sigma (float) : for "Gaussian" kernel 
            a,b (float) : for "Sigmoid" kernel
            h (function) : for "RBF" kernel 
            theta (float) : for "Exponential" kernel

        """
        super(DualRegression,self).__init__(kernel=kernel,sigma=sigma,a=a,b=b,h=h,theta=theta)
        self.lamda = lamda
        self.dual_weight = None 
        self.X = None 

    def fit(self,X,y):
        """fit

        Args:   
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,1) 

        """
        self.X = X 
        N = X.shape[0]
        gram_mat = self.gram_func(X) 
        self.dual_weight = np.dot(np.linalg.inv(gram_mat + self.lamda*np.eye(N)),y)

    def predict(self,X):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)

        Returns:
            y (1-D array) : predicted value, shape = (N_samples,1) 

        """ 
        gram_mat = np.zeros((self.X.shape[0],X.shape[0]))
        for i in range(self.X.shape[0]):
            gram_mat[i] = np.array([self.kernel_func(self.X[i],X[j]) for j in range(X.shape[0])])
        y = gram_mat.T@self.dual_weight
        return y

class NadarayaWatson():
    """NadarayaWatson 

    When X_tr,t is training data, and x is test data, predicted y is expressed as 

        y = \sum_n k(x,X_tr[n])t_n 
        k(x,X_tr[n]) = g(|x-X_tr[n]|) / \sum_m g(|x-X_tr[m]|)
    
    """
    def __init__(self,g):
        """

        Args:   
            g (function) : function for kernel function, g(|x-x_n|)

        """
        self.g = g 
    
    def fit(self,X,y):
        """fit

        Args:   
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,1) 

        """
        self.X = X 
        self.y = y 

    def predict(self,X):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)

        Returns:
            y (2-D array) : predicted value, shape = (N_samples,1) 

        """ 
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            kernel = np.array([self.g(np.dot(X[i]-self.X[j],X[i]-self.X[j])**0.5) for j in range(self.X.shape[0])])
            kernel /= kernel.sum()
            y[i] = np.dot(kernel,self.y)
        
        return y.reshape(-1,1)

class GaussianProcessRegression(BaseKernelMachine):
    """GaussianProcessregression

    Attributes:
        kernel_func (function) : kernel function k(x,y) 
        gram_func (function) : function which make gram matrix 
        alpha,beta (float) : hyperparameter 
        C_inv (2-D array) : 

    """
    def __init__(self,alpha=1.0,beta=5.0,kernel="Linear",sigma=0.1,a=1.0,b=0.0,h=None,theta=1.0):
        """

        Args:
            alpha,beta (float) : hyperparameter 
            kernel (string) : kernel type (default "Linear"). you can choose "Linear","Gaussian","Sigmoid","RBF","Exponential"
            sigma (float) : for "Gaussian" kernel 
            a,b (float) : for "Sigmoid" kernel
            h (function) : for "RBF" kernel 
            theta (float) : for "Exponential" kernel

        """
        super(GaussianProcessRegression,self).__init__(kernel=kernel,sigma=sigma,a=a,b=b,h=h,theta=theta)
        self.alpha = alpha 
        self.beta = beta 

    def fit(self,X,y):
        """fit

        Args:   
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,1) 

        """
        self.X = X 
        self.y = y 
        C = self.gram_func(X)/self.alpha + np.eye(X.shape[0])/self.beta 
        self.C_inv = np.linalg.inv(C) 

    def predict(self,X,return_std=False):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)
            return_std (bool) : if True,also return std of predicted value 

        Returns:
            y (2-D array) : predicted value, shape = (N_samples,1) 
            std (2-D array) : std of predicted value, shape = (N_samples,1)

        """ 
        gram_mat = np.zeros((self.X.shape[0],X.shape[0]))
        for i in range(self.X.shape[0]):
            gram_mat[i] = np.array([self.kernel_func(self.X[i],X[j]) for j in range(X.shape[0])])
        
        y = gram_mat.T@self.C_inv@self.y 
        if return_std:
            c = np.array([self.kernel_func(X[i],X[i]) for i in range(X.shape[0])]).reshape(-1,1) + 1/self.beta 
            sigma = c - np.diag(gram_mat.T@self.C_inv@gram_mat).reshape(-1,1)
            return y,sigma**0.5
        else:
            return y 


class GaussianProcessClassifier(BaseKernelMachine,Classifier):
    """GaussianProcessClassifier

    Attributes:
        kernel_func (function) : kernel function k(x,y) 
        gram_func (function) : function which make gram matrix 
        alpha (float) : hyperparameter
        gamma (float) : noise parameter to ensure C is positive definite  

    """
    def __init__(self,alpha=1.0,gamma=0.1,max_iter=100,threshold=1e-2,kernel="Linear",sigma=0.1,a=1.0,b=0.0,h=None,theta=1.0):
        """

        Args:
            alpha (float) : hyperparameter 
            gamma (float) : noise parameter to ensure C is positive definite  
            max_iter (int) : max iteration for parameter optimization
            threshold (float) : threshold for optimizint parameters 
            kernel (string) : kernel type (default "Linear"). you can choose "Linear","Gaussian","Sigmoid","RBF","Exponential"
            sigma (float) : for "Gaussian" kernel 
            a,b (float) : for "Sigmoid" kernel
            h (function) : for "RBF" kernel 
            theta (float) : for "Exponential" kernel

        """
        super(GaussianProcessClassifier,self).__init__(kernel=kernel,sigma=sigma,a=a,b=b,h=h,theta=theta)
        Classifier.__init__(self) # this part should be fixed 
        self.alpha = alpha 
        self.gamma = gamma 
        self.max_iter = max_iter 
        self.threshold = threshold 


    def fit(self,X,y):
        """fit

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  

        """

        y = self._onehot_to_label(y)
        y = y.reshape(-1,1) 

        self.X = X 
        self.y = y 
        C = self.gram_func(X)/self.alpha + np.eye(X.shape[0])*self.gamma  
        C_inv = np.linalg.inv(C) 

        # search mode using Neweton-method 
        a = np.random.randn(X.shape[0],1)
        for _ in range(self.max_iter):
            sig = sigmoid(a) 
            W = sig*(1 - sig)
            a = C@np.linalg.inv(np.eye(X.shape[0]) + W*C)@(y - sig + W*a) 
            da = y - sig - C_inv@a 
            if np.dot(da.ravel(),da.ravel())**0.5 < self.threshold:
                break
        
        self.C = C 
        self.a = a
        
    def predict(self,X,return_prob=False):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
            return_prob (bool) : if True, return probability 

        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 
            or if return_prob == True
            y (1-D array) :  always return probability of belonging to class1 in each record 

        """
        gram_mat = np.zeros((self.X.shape[0],X.shape[0]))
        for i in range(self.X.shape[0]):
            gram_mat[i] = np.array([self.kernel_func(self.X[i],X[j]) for j in range(X.shape[0])]) 

        sig = sigmoid(self.a)
        logit = (gram_mat.T@(self.y - sig)).ravel()
        sig = sig.ravel()

        if return_prob:
            c = np.array([self.kernel_func(X[i],X[i]) for i in range(X.shape[0])]) + self.gamma
            W = np.diag(1/(sig*(1 - sig))) 
            sigma = c - np.diag(gram_mat.T@np.linalg.inv(W + self.C)@gram_mat)
            prob = sigmoid(kappa(sigma)*logit) 
            return prob  
        else: 
            y = np.zeros(X.shape[0])
            y[logit >= 0] = 1
            return self._inverse_transform(y)  