"""Datasets

"""

import numpy as np 

class RegressionDataGenerator():
    """RegressionDataGenerator

    Create 1-D toy data for regression 
    """
    def __init__(self,f):
        """
        Args:
            f (object) : generate 1-D data which follows f(x) + gauss noise 
        """
        self.f = f 
    
    def __call__(self,n = 50,lower = 0,upper = 2*np.pi,std = 1): 
        """Make data 
        Args:
            n (int) : number of data 
            lower,upper (float) : generate data lowe <= x <= upper 
            std (float) : std of gauss noise
        Returns:
            X (2-D array) : explanatory variable,shape = (N_samples,1)
            y (2-D array) : target variable, shape = (N_samples,1) 
        """
        X = np.random.rand(n)*(upper-lower) + lower
        y = self.f(X) + np.random.randn(n)*std 
        return X.reshape(-1,1),y.reshape(-1,1)
