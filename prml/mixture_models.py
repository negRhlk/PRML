"""MIxuture Models 

chapter9
KMeans 
GaussianMixture 
BernoulliMixture 

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
        K (int): number of cluster 
        max_iter (int): number of max iteration 
        threshold (float): threshold 

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
    
    def fit(self,X,init_gamma=None,init_pi=None,init_mu=None,init_sigma=None):
        """fit 

        N := N_samples 
        M := N_dim 
        K := number of mixture 

        Args:
            X (2-D array): shape = (N,M), data
            init_gamma (2-D array): shape = (N,K), initial responsibility
            init_pi (1-D array): shape = (K), initial pi 
            init_mu (2-D array): shape = (K,M), initial mus
            init_sigma (3-D array): shape = (K,M,M), initial sigmas
        
        Returns:
            pi (1-D array): shape = (K) 
            mu (2-D array): shape = (K,M) 
            sigma (3-D array): shape = (K,M,M)

        """

        N = X.shape[0] 
        M = X.shape[1]

        gamma = init_gamma
        pi = init_pi 
        mu = init_mu
        sigma = init_sigma 

        if gamma is None: 
            gamma = np.random.rand(N,self.K)
            gamma /= gamma.sum(axis = 1,keepdims=True)
        
        if pi is None:
            pi = np.zeros(self.K)

        if mu is None:
            mu = np.zeros((self.K,M)) 
        
        if sigma is None:
            sigma = np.array([np.eye(M) for _ in range(self.K)])
        
        before_log_likelihood = -np.Inf 

        for _ in range(self.max_iter):

            # M step 
            N_k = gamma.sum(axis = 0)
            mu = (gamma.T@X) / N_k.reshape(-1,1) 
            tmp = X.reshape(N,1,M) - mu.reshape(1,self.K,M)
            sigma = np.sum(gamma.reshape(N,self.K,1,1)*(tmp.reshape(N,self.K,M,1)*tmp.reshape(N,self.K,1,M)),axis = 0) / N_k.reshape(-1,1,1)
            pi = N_k/N

            # E step 
            tmp = X.reshape(N,1,M) - mu.reshape(1,self.K,M)
            normalize_and_pi = (2*np.pi)**(-M/2) * np.linalg.det(sigma)**(-0.5) * pi 
            gauss = normalize_and_pi * np.exp( -0.5 * tmp.reshape(N,self.K,1,M)@sigma.reshape(1,self.K,M,M)@tmp.reshape(N,self.K,M,1)).reshape(N,self.K) 

            log_likelihood = np.sum(np.log(gauss.sum(axis = 1))) 
            if abs(before_log_likelihood - log_likelihood) < self.threshold:
                break 
            before_log_likelihood = log_likelihood

            gamma = gauss / gauss.sum(axis = 1,keepdims=True) 
        return pi,mu,sigma


class BernoulliMixture():
    """BernoulliMixture

    Attributes:
        K (int): number of cluster 
        max_iter (int): number of max iteration 
        threshold (float): threshold 

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

    def fit(self,X,init_gamma=None,init_pi=None,init_mu=None):
        """fit 

        N := N_samples 
        M := N_dim 
        K := number of mixture 

        Args:
            X (2-D array): shape = (N,M), data, value is 0 or 1
            init_gamma (2-D array): shape = (N,K), initial responsibility
            init_pi (1-D array): shape = (K), initial pi 
            init_mu (2-D array): shape = (K,M), initial mus

        Returns:
            pi (1-D array): shape = (K) 
            mu (2-D array): shape = (K,M) 

        """

        N = X.shape[0] 
        M = X.shape[1]

        gamma = init_gamma
        pi = init_pi 
        mu = init_mu

        if gamma is None: 
            gamma = np.random.rand(N,self.K) + 0.10
            gamma /= gamma.sum(axis = 1,keepdims=True)
        
        if pi is None:
            pi = np.zeros(self.K)

        if mu is None:
            mu = np.zeros((self.K,M)) 
        
        before_log_likelihood = -np.inf
        for _ in range(self.max_iter):

            # M step 
            N_k = gamma.sum(axis = 0)
            mu = (gamma.T@X) / N_k.reshape(-1,1)
            pi = N_k/N

            # E step 
            tmp = mu.reshape(1,self.K,M)**X.reshape(N,1,M) * (1.0 - mu.reshape(1,self.K,M))**(1.0 - X.reshape(N,1,M))
            bern = np.prod(tmp,axis = 2)*pi 

            log_likelihood = np.sum(np.log(bern.sum(axis = 1))) 
            if abs(before_log_likelihood - log_likelihood) < self.threshold:
                break 
            before_log_likelihood = log_likelihood

            gamma = bern / bern.sum(axis = 1,keepdims=True)

        return pi,mu 