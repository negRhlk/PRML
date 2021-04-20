"""
Design Matrix 

Basis functions are
    - gauss
    - sigmoid
    - polynomial
"""
import numpy as np 
from abc import abstractclassmethod,ABCMeta


class DesignMat(metaclass=ABCMeta):
    """DesignMat Base Class 

    Attributes: 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter 
    """
    def __init__(self,mu = None,s = None):
        if mu is None:
            self.mu = np.random.normal(loc = 0,scale = 10,size = 10)
        else:
            self.mu = mu 
        if s is None:
            s = np.random.normal(loc = 0,scale = 10,size = 10)
            s[abs(s) < 1e-10] = 1e-5
            self.s = s
        else:
            self.s = s 

    @abstractclassmethod
    def _basis_function(self,x):
        """Basis function 
        
        Args:
            x (1-D array) : data,shape = (N_dim) 
        
        Returns :
            phi (1-D array) : data.shape = (N_featuredim)
        """
        pass 

    def __call__(self,X):
        """Make design matrix using X

        Args :
            X (2-D array) : data, shape = (N_samples,N_dim) 

        Returns : 
            design_mat (2-D array) : design_mat, shape = (N_samples,N_featuredim)
        """
        return np.vstack([self._basis_function(x) for x in X])


class GaussMat(DesignMat):
    """Gauss Basis Function 

    Attributes: 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter
    """
    def __init__(self,mu = None,s = None):
        super(GaussMat,self).__init__(mu,s) 
    
    def _basis_function(self,x):
        """Basis function 
        
        Args:
            x (1-D array) : data,shape = (N_dim) 
        
        Returns :
            phi (1-D array) : data.shape = (N_featuredim)
        """
        phi = np.exp(-(x.reshape(-1,1) - self.mu)/(2*self.s**2))
        phi = np.concatenate(([1],phi.ravel()))
        return phi


class SigmoidMat(DesignMat):
    """Sigmoid Basis Function 

    Attributes: 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter
    """
    def __init__(self,mu = None,s = None):
        super(SigmoidMat,self).__init__(mu,s) 
    
    def _basis_function(self,x):
        """Basis function 
        
        Args:
            x (1-D array) : data,shape = (N_dim) 
        
        Returns :
            phi (1-D array) : data.shape = (N_featuredim)
        """
        a = (x.reshape(-1,1) - self.mu)/self.s
        a = a.ravel() 
        phi = np.zeros(a.shape[0])
        phi[a >= 0] = 1/(1 + np.exp(-a[a >= 0])) 
        phi[a < 0] = np.exp(a[a < 0])/(np.exp(a[a < 0]) + 1) 
        return np.concatenate(([1],phi)) 

class PolynomialMat(DesignMat):
    """Polynomial Basis Function 

    Attributes:
        deg (int) : max degree of polynomial features
    """
    def __init__(self,deg = None):
        super(PolynomialMat,self).__init__() 
        if deg is None:
            self.deg = 5
        else:
            self.deg = deg 
    
    def _basis_function(self,x):
        """Basis function 
        
        Args:
            x (1-D array) : data,shape = (N_dim) 
        
        Returns :
            phi (1-D array) : data.shape = (N_featuredim)
        """
        phi = [np.array([1])]
        for n in range(1,self.deg+1): 
            deg_n_feature = np.power(x,n) 
            phi.append(deg_n_feature)
        return np.concatenate(phi) 


