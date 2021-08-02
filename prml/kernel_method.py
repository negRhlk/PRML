"""kernel methods 

Chapter6 

"""

import numpy as np 
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


        

