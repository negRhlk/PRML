"""sequential data 
    AR
    
"""

import numpy as np 
from abc import ABC,abstractclassmethod, abstractstaticmethod

from prml.utils.encoder import OnehotToLabel
from prml.linear_regression import LinearRegression

class AR():
    """AR

    auto regressive model 

    """
    def __init__(self,p) -> None:
        """

        Args:
            p (int): number of past data used to predict next variables 

        """
        self.p = p
    
    def _make_data(self,X):
        N = X.shape[0] - self.p 
        data = np.zeros((N,self.p))
        for i in range(N):
            data[i] = X[i:i+self.p]
        return data 

    def fit(self,X):
        """fit 

        Args:
            X (1-D array): sequential data

        """

        y = X[self.p:] 
        X = self._make_data(X)
        model = LinearRegression(basis_function="polynomial",deg=1)
        model.fit(X,y.reshape(-1,1))
        self.model = model 
    
    def predict(self,X,size=1):
        """predict

        Args:
            X (1-D array): sequential data 
            size (int): number of predict data from the lasta data of X 
        
        Returns:
            pred (1-D array): shape = (size), predicted variables 

        """
        X = X[-self.p:]
        pred = np.zeros(size)
        for i in range(size):
            y = self.model.predict(X.reshape(1,-1)).ravel()
            pred[i] = y[0]  
            X = np.concatenate((X[1:],y))
        
        return pred 


class BaseHMM(ABC):
    """BaseHMM

    Attributes:

    """
    def __init__(self,K,max_iter=100,threshold=1e-5) -> None:
        """init

        Args:
            K (int): dimension of latent space
            max_iter (int): number of max iteration 
            threshold (float): threshold 

        """
        self.K = K 
        self.max_iter = max_iter 
        self.threshold = threshold
    
    def _Estep(self,X,pi,A,params):
        """_Estep 

        forward backward algotithum

        Args:
            X (2-D array): shape = (N,N_dim),time series data
            pi (1-D array): pi[i]
            A (2-D array): A[j][k] = p(z_k=1|z_j=1)
            params (object): params 

        """

        N = X.shape[0] 
        pXZ = self._prob_cond(X,params)

        # alpha 
        alpha = np.zeros((N,self.K))
        alpha[0] = pi*pXZ[0]
        for i in range(N-1):
            alpha[i+1] = pXZ[i+1]*np.dot(A.T,alpha[i])

        # beta 
        beta = np.zeros((N,self.K)) 
        beta[N-1] = np.ones(self.K) 
        for i in range(N-2,-1,-1):
            beta[i] = np.dot(A,beta[i+1]*pXZ[i+1])

        likelihood = np.sum(alpha[-1])
        gamma = alpha*beta/likelihood
        xi = alpha[:-1].reshape(N-1,self.K,1)*pXZ[1:].reshape(N-1,1,self.K)*A*beta[1:].reshape(N-1,1,self.K)/likelihood
        return gamma,xi,likelihood

    def _Mstep(self,X,gamma,xi):
        """Mstep

        Args:
            X (2-D array): shape = (N,N_dim),time series data
            gamma (2-D array): shape = (N,K), probablity of belonging to certain latent space
            xi (3-D array): shape = (N-1,K,K), transition probablity 

        """
        pi = gamma[0]/gamma[0].sum()
        A = xi.sum(axis = 0)/xi.sum(axis = (0,2)) 
        params = self._specializeMstep(X,gamma)
        return pi,A,params  
    
    def fit(self,X):
        """fit

        Args:
            X (2-D array): shape = (N,N_dim),time series data

        """

        N = X.shape[0] 
        gamma = np.random.rand(N,self.K)
        xi = np.random.rand(N-1,self.K,self.K)

        for _ in range(self.max_iter):
            pi,A,params = self._Mstep(X,gamma,xi)
            gamma_new,xi_new,likelihood = self._Estep(X,pi,A,params)

            if np.mean((gamma_new - gamma)**2)**0.5 + np.mean((xi_new - xi)**2)**0.5 < self.threshold:
                gamma = gamma_new
                xi = xi_new 
                break 

            gamma = gamma_new
            xi = xi_new 
        
        self.pi = pi 
        self.A = A 
        self.params = params 

    def predict(self,X,size=1):
        """predict

        Args:
            X (2-D array): shape = (N,N_dim),time series data
            size (int): number of predict data from the lasta data of X 
        
        Returns:
            pred (2-D array): shape = (size,-1),predicted variables 

        """
        pred = np.zeros((size,X.shape[1]))
        gamma,_,_ = self._Estep(X,self.pi,self.A,self.params)
        z = gamma[-1]
        for i in range(size):
            z = np.dot(self.A.T,z)
            pred[i] = self._predict(z)
        return pred 
    
    @abstractclassmethod
    def _specializeMstep(self,X,gamma):
        """_specializeMstep 

        Args:
            X (2-D array): shape = (N,N_dim),time series data
            gamma (2-D array): shape = (N,K), probablity of belonging to certain latent space
        
        Returns:
            params (object): optimized param 

        """
        pass 

    @abstractclassmethod
    def _prob_cond(self,X,params):
        """

        px[i] express p(X|z_k=1)

        Args:
            X (2-D array): shape = (N,N_dim)
            params (object): param 
        
        Returns:
            pX (2-D array): shape = (N,K),probablity of X

        """
        pass 

    @abstractclassmethod
    def _predict(self,z):
        """_predict 

        Args:
            z (1-D array): shape = (K),latent variables 

        Returns:
            Ex (1-D array): shape = (N_dim),expected value of X under the conditions of z
        """
        pass 


class GaussHMM(BaseHMM):
    """GaussHMM

    Attributes:

    """
    def __init__(self, K, max_iter=100, threshold=1e-5) -> None:
        super(GaussHMM,self).__init__(K, max_iter=max_iter, threshold=threshold)
    
    def _specializeMstep(self,X,gamma):
        """_specializeMstep 

        Args:
            X (2-D array): shape = (N,N_dim),time series data
            gamma (2-D array): shape = (N,K), probablity of belonging to certain latent space 
        
        Returns:
            params (object): optimized param 

        """
        
        N = X.shape[0] 
        mu = gamma.T@X/gamma.sum(axis = 0).reshape(-1,1)
        tmp = X.reshape(N,1,-1) - mu.reshape(1,self.K,-1)
        sigma = np.sum(gamma.reshape(N,self.K,1,1)*tmp.reshape(N,self.K,-1,1)*tmp.reshape(N,self.K,1,-1),axis=0)/gamma.sum(axis=0).reshape(self.K,1,1)
        return {
            "mu":mu,
            "sigma":sigma,
            "sigma_inv":np.linalg.inv(sigma)
        }

    def _prob_cond(self,X,params):
        """

        px[i] express p(X|z_k=1)

        Args:
            X (2-D array): shape = (N,N_dim)
            params (object): param 
        
        Returns:
            pX (2-D array): shape = (N,K),probablity of X

        """
        norm_const = 1/((2*np.pi)**X.shape[1]*np.linalg.det(params["sigma"]))**0.5
        N_dim = X.shape[1]
        tmp = X.reshape(-1,1,N_dim) - params["mu"].reshape(1,self.K,N_dim)
        pX = norm_const*np.exp(-(tmp.reshape(-1,self.K,N_dim,1)@params["sigma_inv"]@tmp.reshape(-1,self.K,1,N_dim)).reshape(-1,self.K))
        return pX 

    def _predict(self,z):
        """_predict 

        Args:
            z (1-D array): shape = (K),latent variables 

        Returns:
            Ex (1-D array): shape = (N_dim),expected value of X under the conditions of z
        """
        return np.dot(self.params["mu"].T,z)


class BernoulliHMM(BaseHMM):
    """BernoulliHMM

    Attributes:

    """
    def __init__(self, K, max_iter=100, threshold=1e-5) -> None:
        super().__init__(K, max_iter=max_iter, threshold=threshold)
    
    def _specializeMstep(self,X,gamma):
        """_specializeMstep 

        Args:
            X (2-D array): shape = (N,N_dim),time series data
            gamma (2-D array): shape = (N,K), probablity of belonging to certain latent space
        
        Returns:
            params (object): optimized param 

        """
        mu = gamma.T@X/gamma.sum(axis = 0).reshape(-1,1)
        return {
            "mu":mu
        }

    def _prob_cond(self,X,params):
        """

        px[i] express p(X|z_k=1)

        Args:
            X (2-D array): shape = (N,N_dim)
            params (object): param 
        
        Returns:
            pX (2-D array): shape = (N,K),probablity of X

        """
        ohe = OnehotToLabel() 
        y = ohe.fit_transform(X).astype("int")
        return params["mu"][:,y].T

    def _predict(self,z):
        """_predict 

        Args:
            z (1-D array): shape = (K),latent variables 

        Returns:
            Ex (1-D array): shape = (N_dim),expected value of X under the conditions of z
        """
        return np.dot(self.params["mu"].T,z)