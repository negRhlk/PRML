"""Continuous Latent Variables 

"""

import numpy as np 

class PCA():
    """PCA
    """
    def __init__(self):
        pass
    
    def fit(self,X):
        """fit 

        Args:
            X (2-D array): shape = (N_samples,N_dim), data

        """

        N = X.shape[0]
        X_mean = X.mean(axis = 0) 
        S = (X - X_mean).T@(X - X_mean)/N 
        eig_val,eig_vec = np.linalg.eig(S)
        eig_val,eig_vec = np.real(eig_val),np.real(eig_vec.real)

        self.importance = eig_val/eig_val.sum()
        self.weight = eig_vec
    
    def transform(self,X,M,return_importance=False):
        """transform

        Args:
            X (2-D array): shape = (N_samples,N_dim), data
            M (int): number of principal component, if M > N_dim, M = N_dim 
            return_importance (bool): return importance or not
        
        Retunrs:
            X_proj (2-D array): shape = (N_samples,N_dim), projected data
            impotance_rate (float): how important X_proj is

        """
        return X@self.weight[:,:M],self.importance[:M].sum()
    
    def fit_transform(self,X,M,return_importance=False):
        """fit_transform

        Args:
            X (2-D array): shape = (N_samples,N_dim), data
            M (int): number of principal component, if M > N_dim, M = N_dim 
            return_importance (bool): return importance or not
        
        Retunrs:
            X_proj (2-D array): shape = (N_samples,N_dim), projected data
            impotance_rate (float): how important X_proj is

        """
        self.fit(X)
        return self.transform(X,M,return_importance)



