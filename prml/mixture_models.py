"""MIxuture Models 

    KMeans 

"""

import numpy as np 


class KMeans():
    """KMeans 

    Attributes:
        K (int): number of cluster 
            ax_iter (int): number of max iteration 

    """
    def __init__(self,K,max_iter=100):
        """

        Args:
            K (int): number of cluster 
            max_iter (int): number of max iteration 

        """
        self.K = K 
        self.max_iter = max_iter

    def fit(self,X,init_prototype=None):
        """

        Args:
            X (2-D array): shape = (N_samples,N_dims)
        
        Returns:
            cluster (1-D array): shape = (N_samples), index of group which a record belongs to
            prototype (2-D array): shape = (K,N_dims), prototype vector

        """

        if init_prototype is None:
            init_prototype = np.random.randn(self.K,X.shape[1]) 
        prototype = init_prototype
        cluster = np.zeros(X.shape[0])
        
        for _ in range(self.max_iter):
            changed = 0

            # E step 
            for i in range(X.shape[0]):
                dist = np.sum((X[i] - prototype)**2,axis = 1) 
                min_dist_idx = np.argmin(dist)  
                if cluster[i] != min_dist_idx:
                    changed += 1
                cluster[i] = min_dist_idx
            
            # M step
            for i in range(self.K):
                if np.all(cluster != i):
                    continue
                prototype[i] = np.mean(X[cluster == i],axis = 0)
            
            if changed == 0:
                break 
        
        return cluster,prototype


class GaussianMixture():
    """GaussianMixture 

    Attributes:

    """
    def __init__(self,K,max_iter=100,threshold=1e-7):
        """

        Args:
            K (int): number of cluster 
            max_iter (int): number of max iteration 
            threshold (float): threshold 

        """
        self.K = K 
        self.max_iter = max_iter 
        self.threshold = threshold
    
    def fit(self,X,gamma=None):
        """fit 

        Args:

        """

        N = X.shape[0] 
        M = X.shape[1]
        
        if gamma is None: 
            gamma = np.random.randn(N,self.K)
            gamma /= gamma.sum(axis = 1,keepdims=True)
    
        mu = np.zeros((self.K,M)) 
        sigma = np.zeros((self.K,M,M)) 
        pi = np.zeros(self.K)
        
        for _ in range(self.max_iter):

            # M step 
            N_k = gamma.sum(axis = 0)
            new_mu = np.sum(gamma.reshape(N,self.K,1)*X.reshape(N,1,M),axis = 0) / N_k.reshape(-1,1)
            tmp = X.reshape(N,1,M) - mu.reshape(1,self.K,M)
            new_sigma = np.sum(gamma.reshape(N,self.K,1,1)*(tmp.reshape(N,self.K,M,1)*tmp.reshape(N,self.K,1,M)),axis = 0) / N_k.reshape(-1,1)
            new_pi = N_k/N

            change = np.mean((new_mu - mu)**2) + np.mean((new_sigma - sigma)**2) + np.mean((new_pi - pi)**2) 
            if change**0.5 < self.threshold:
                break 

            mu = new_mu 
            sigma = new_sigma
            pi = new_pi

            # E step 
            tmp = X.reshape(N,1,M) - mu.reshape(1,self.K,M)
            normalize_and_pi = (2*np.pi)**(-M/2) * np.linalg.det(sigma)**(-0.5) * pi 
            gauss = normalize_and_pi * np.exp( -0.5 * tmp.reshape(N,self.K,1,M)@sigma.reshape(1,self.K,M,M)@tmp.reshape(N,self.K,M,1)).reshape(N,self.K) 
            gamma = gauss / gamma.sum(axis = 1,keepdims=True) 
        
        return pi,mu,sigma