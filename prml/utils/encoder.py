"""
Encoder 

- transform onehot-encoded data to label-encoded data
- transform label-encoded data to onehot-encoded data
"""

import numpy as np 

class OnehotToLabel():
    """Transform onehot-encoded data to label-encoded data

    Attributes:
        K (int) : number of class 
    """
    def __init__(self):
        pass 
    def fit(self,X):
        """fit
        Args:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class) 
        """
        self.K = X.shape[1] 

    def transform(self,X):
        """transform 
        Args:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class) 
        Returns:
            y (1-D array) : label-encoded data, shape = (N_samples)
        """
        y = np.zeros(X.shape[0])
        for k in range(self.K): 
            y[X[:,k] == 1] = k 
        return y 
    def fit_transform(self,X):
        """fit and transform 
        Args:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class) 
        Returns:
            y (1-D array) : label-encoded data, shape = (N_samples)
        """
        self.fit(X)
        return self.transform(X) 
    def inverse(self,y): 
        """inverse transform 
        Args:
            y (1-D array) : label-encoded data, shape = (N_samples) 
        Returns:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class)
        """
        X = np.zeros((y.shape[0],self.K)) 
        for k in range(self.K):
            X[y == k,k] = 1 
        return X  

class LabelToOnehot(): 
    """Transform label-encoded data to onehot-encoded data

    Attributes:
        K (int) : number of class 
        data_group (1-D array) : data which is labeled 
    """
    def __init__(self):
        pass 

    def fit(self,y):
        """fit
        Args:
            y (1-D array) : label-encoded data, shape = (N_samples) 
        """
        self.data_group = np.unique(y)  
        self.K = len(self.data_group)
    
    def transform(self,y):
        """transform 
        Args:
            y (1-D array) : label-encoded data, shape = (N_samples) 
        Returns:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class)
        """
        X = np.zeros((y.shape[0],self.K)) 
        for k,g in enumerate(self.data_group):
            X[y == g,k] = 1 
        return X  

    def fit_transform(self,y):
        """fit and transform 
        Args:
            y (1-D array) : label-encoded data, shape = (N_samples) 
        Returns:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class)
        """
        self.fit(y)
        return self.transform(y) 

    def inverse(self,X): 
        """inverse transform 
        Args:
            X (2-D array) : onehot-encoded data, shape = (N_samples,N_class) 
        Returns:
            y (1-D array) : label-encoded data, shape = (N_samples)
        """
        y = np.zeros(X.shape[0])
        for k,g in enumerate(self.data_group): 
            y[X[:,k] == 1] = g 
        return y 