"""Kernel function 

LinearKernel 
GaussianKernel 
SigmoidKernel
RBFKernel 
ExponentialKernel

GramMatrix

"""

import numpy as np 
from abc import ABC,abstractclassmethod

class BaseKernel(ABC):
    def __init__(self):
        pass 

    @abstractclassmethod
    def __call__(self,x,y):
        """

        Args:
            x,y (1-D array) : calculate k(x,y)
        
        Returns:    
            k (float) : k(x,y) 

        """
        pass 

class LinearKernel(BaseKernel):
    """Linear Kernel 

    Linear Kernel : k(x,y) = x^Ty

    """
    def __init__(self):
        super(LinearKernel,self).__init__()

    def __call__(self, x, y):
        """

        Args:
            x,y (1-D array) : calculate k(x,y)
        
        Returns:    
            k (float) : k(x,y) = x^Ty

        """
        return np.dot(x,y)

class GaussianKernel(BaseKernel):
    """Gaussian Kernel 

    Gaussian Kernel : k(x,y) = exp(-|x-y|^2/2sigma^2)

    """
    def __init__(self,sigma=0.1):
        """

        Args:   
            sigma (float) : parameter for kernel function 

        """
        super(GaussianKernel,self).__init__()
        self.sigma = sigma 

    def __call__(self, x, y):
        """

        Args:
            x,y (1-D array) : calculate k(x,y)
        
        Returns:    
            k (float) : k(x,y) = exp(-|x-y|^2/2sigma^2)

        """
        return np.exp(-np.dot(x-y,x-y)/(2*self.sigma**2))

class SigmoidKernel(BaseKernel):
    """Sigmoid Kernel

    Sigmoid Kernel: k(x,y) = tanh(ax^Ty + b)

    """
    def __init__(self,a=1,b=0):
        """

        Args:   
            a,b (float) : parameter for kernel function

        """
        super(SigmoidKernel,self).__init__()
        self.a = a 
        self.b = b 
    
    def __call__(self, x, y):
        """

        Args:
            x,y (1-D array) : calculate k(x,y)
        
        Returns:    
            k (float) : k(x,y) = tanh(ax^Ty + b)

        """
        return np.tanh(self.a*np.dot(x,y) + self.b)

class RBFKernel(BaseKernel):
    """RBF Kernel 

    RBF Kernel : k(x,y) = h(|x - y|) 

    """
    def __init__(self,h):
        """

        Args:
            h (function) : k(x,y) = h(|x-y|)

        """
        super(RBFKernel,self).__init__()
        self.h = h 
    
    def __call__(self, x, y):
        """

        Args:
            x,y (1-D array) : calculate k(x,y)
        
        Returns:    
            k (float) : k(x,y) = h(|x - y|) 

        """
        return self.h(np.dot(x-y,x-y)**0.5)

class ExponentialKernel(BaseKernel):
    """Exponential Kernel 

    Exponential Kernel : k(x,y) = exp(-theta|x - y|)

    """
    def __init__(self,theta=1):
        """

        Args:   
            theta (float) : parameter for kernel function

        """
        super(ExponentialKernel,self).__init__()
        self.theta = theta 
    
    def __call__(self, x, y):
        """

        Args:
            x,y (1-D array) : calculate k(x,y)
        
        Returns:    
            k (float) : k(x,y) = exp(-theta|x - y|)

        """
        return np.exp(-self.theta*np.dot(x-y,x-y)**0.5)


class GramMatrix():
    """Gram Matrix 

    For making gram matrix.
    kernel type is 
        LinearKernel 
        GaussianKernel 
        SigmoidKernel
        RBFKernel 
        ExponentialKernel 
    

    Attributes: 
        kernel_func (function) : kernel_function
    
    """
    def __init__(self,kernel="Linear",sigma=0.1,a=1.0,b=0.0,h=None,theta=1.0):
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
    
    def __call__(self,X):
        """

        Args:   
            X (2-D array) : shape = (N_samples,N_dims)
        
        Returns:
            G (2-D array) : shape = (N_samples,N_samples) 
        
        Note:
            time complexity is O(N_samples^2*N_dims)

        """
        N = X.shape[0] 
        G = np.zeros((N,N))
        for i in range(N):
            G[i] = np.array([self.kernel_func(X[i],X[j]) for j in range(N)])
        return G 