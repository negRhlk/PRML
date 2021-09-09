"""Continuous Latent Variables 

    PCA
    ProbabilisticPCA
    
"""

import numpy as np 

class PCA():
    """PCA

    Attributes:
        X_mean (1-D array): mean of data
        weight (2-D array): proj matrix 
        importance (1-D array): contirbution of ratio

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
        eig_val,eig_vec = np.linalg.eigh(S)
        eig_val,eig_vec = np.real(eig_val),np.real(eig_vec.real)
        idx = np.argsort(eig_val)[::-1]
        eig_val,eig_vec = eig_val[idx],eig_vec[:,idx]

        self.X_mean = X_mean 
        self.importance = eig_val/eig_val.sum()
        self.weight = eig_vec
    
    def transform(self,X,M,return_importance=False,whitening=False):
        """transform

        Args:
            X (2-D array): shape = (N_samples,N_dim), data
            M (int): number of principal component, if M > N_dim, M = N_dim 
            return_importance (bool): return importance or not
            whitening (bool): if whitening or not
        
        Retunrs:
            X_proj (2-D array): shape = (N_samples,M), projected data
            impotance_rate (float): how important X_proj is

        """
        if whitening:
            return (X-self.X_mean)@self.weight[:,:M]/np.sqrt(self.importance[:M])
        elif return_importance:
            return X@self.weight[:,:M],self.importance[:M].sum()
        else:
            return X@self.weight[:,:M]

    def fit_transform(self,X,M,return_importance=False,whitening=False):
        """fit_transform

        Args:
            X (2-D array): shape = (N_samples,N_dim), data
            M (int): number of principal component, if M > N_dim, M = N_dim 
            return_importance (bool): return importance or not
            whitening (bool): if whitening or not
        
        Retunrs:
            X_proj (2-D array): shape = (N_samples,M), projected data
            impotance_rate (float): how important X_proj is

        """
        self.fit(X)
        return self.transform(X,M,return_importance,whitening)


class ProbabilisticPCA():
    """ProbabilisticPCA

    find parameter by maximum likelihood method, O(D^3)

    Attributes:
        D (int): original dim of data
        mu (1-D array): mean of data
        W (2-D array): param of density of data
        sigma (float): param of density of data
        U (2-D array): eigen vectors of covariance matrix of data
        lamda (1-D array): eigen values of covariance matrix of data

    """
    def __init__(self) -> None:
        pass

    def fit(self,X):
        """

        Args:
            X (2-D array): shape = (N_samples,N_dim), data

        """

        N = X.shape[0]
        X_mean = X.mean(axis = 0) 
        S = (X - X_mean).T@(X - X_mean)/N 
        eig_val,eig_vec = np.linalg.eigh(S)
        eig_val,eig_vec = np.real(eig_val),np.real(eig_vec.real)
        idx = np.argsort(eig_val)[::-1]
        eig_val,eig_vec = eig_val[idx],eig_vec[:,idx]

        self.D = X.shape[1]
        self.mu = X_mean
        self.U = eig_vec 
        self.lamda = eig_val 

    def transform(self,X,M):
        """transform 

        after this method is called, attribute W,sigma can be used

        Args:
            X (2-D array): shape = (N_samples,N_dim), data
            M (int): number of principal component, M is less than X.shape[1]
        
        Returns:
            X_proj (2-D array): shape = (N_samples,M), projected data

        """
        if self.D == M:
            raise ValueError("M is less than X.shape[1]")
        
        sigma = np.mean(self.lamda[M:])
        W = self.U[:,:M]@(np.diag((self.lamda[:M] - sigma)**0.5))
        
        Mat = W.T@W + sigma*np.eye(M)
        proj_weight = W@np.linalg.inv(Mat) # x -> z
        return (X - self.mu)@proj_weight

    def fit_transform(self,X,M):
        """fit_transform

        after this method is called, attribute W,sigma can be used

        Args:
            X (2-D array): shape = (N_samples,N_dim), data
            M (int): number of principal component, M is less than X.shape[1]
        
        Returns:
            X_proj (2-D array): shape = (N_samples,M), projected data

        """
        self.fit(X)
        return self.transform(X,M)



# class ProbabilisticPCAbyEM():
#     """ProbabilisticPCAbyEM

#     Attributes:
#         D (int): original dim of data
#         mu (1-D array): mean of data
#         W (2-D array): param of density of data
#         sigma (float): param of density of data
#         U (2-D array): eigen vectors of covariance matrix of data
#         lamda (1-D array): eigen values of covariance matrix of data

#     """
#     def __init__(self) -> None:
#         pass

#     def fit(self,X):
#         """

#         Args:
#             X (2-D array): shape = (N_samples,N_dim), data

#         """

#         N = X.shape[0]
#         X_mean = X.mean(axis = 0) 
#         S = (X - X_mean).T@(X - X_mean)/N 
#         eig_val,eig_vec = np.linalg.eigh(S)
#         eig_val,eig_vec = np.real(eig_val),np.real(eig_vec.real)
#         idx = np.argsort(eig_val)[::-1]
#         eig_val,eig_vec = eig_val[idx],eig_vec[:,idx]

#         self.D = X.shape[1]
#         self.mu = X_mean
#         self.U = eig_vec 
#         self.lamda = eig_val 

#     def transform(self,X,M):
#         """transform 

#         after this method is called, attribute W,sigma can be used

#         Args:
#             X (2-D array): shape = (N_samples,N_dim), data
#             M (int): number of principal component, M is less than X.shape[1]
        
#         Returns:
#             X_proj (2-D array): shape = (N_samples,M), projected data

#         """
#         if self.D == M:
#             raise ValueError("M is less than X.shape[1]")
        
#         sigma = np.mean(self.lamda[M:])
#         W = self.U[:,:M]@(np.diag((self.lamda[:M] - sigma)**0.5))
        
#         Mat = W.T@W + sigma*np.eye(M)
#         proj_weight = W@np.linalg.inv(Mat) # x -> z
#         return (X - self.mu)@proj_weight

#     def fit_transform(self,X,M):
#         """fit_transform

#         after this method is called, attribute W,sigma can be used

#         Args:
#             X (2-D array): shape = (N_samples,N_dim), data
#             M (int): number of principal component, M is less than X.shape[1]
        
#         Returns:
#             X_proj (2-D array): shape = (N_samples,M), projected data

#         """
#         self.fit(X)
#         return self.transform(X,M)

