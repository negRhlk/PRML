"""Approximate Inference 
    ApproximateGauss1D
    ApproximateGaussianMixture
"""

import numpy as np 
from scipy.special import psi as digamma 
from prml.utils.util import _log

class ApproximateGauss1D():
    """ApproximateGauss1D

    approximately predict posterior distribution for gaussian


    """
    def __init__(self,a=0,b=0,mu=0,lamda=0,max_iter=1000,threshold=1e-2):
        """

        Args:
            a,b (float): hyper parameter for prior distibution
            mu,lamda (float): hyper parameter for prior distibution
            max_iter (int): max iteration 
            threshold (float): threshold

        """
        self.a = a 
        self.b = b 
        self.mu = mu 
        self.lamda = lamda 
        self.max_iter = max_iter 
        self.threshold = threshold

    def fit(self,X,init_tau=1.0):
        """

        Args:
            X (1-d array): data 
            init_tau (float): initial value for E_tau 
        
        """
        N = len(X)
        X_sum = X.sum()
        X2_sum = np.sum(X**2) 

        E_tau = init_tau
        mu_N = (self.lamda*self.mu + X_sum)/(self.lamda + N)
        lam_N = (self.lamda + N)*E_tau

        for _ in range(self.max_iter):
            E_mu = mu_N 
            E_mu2 = mu_N**2 + 1/((lam_N + N)*E_tau) 
            a_N = self.a + (N + 1)/2 
            b_N = self.b + 0.5*(X2_sum -2*E_mu*X_sum + N*E_mu2 + self.lamda*(E_mu2 - 2*E_mu*self.mu + self.mu**2)) 

            E_tau = a_N/b_N 
            new_mu_N = (self.lamda*self.mu + X_sum)/(self.lamda + N)
            new_lam_N = (self.lamda + N)*E_tau

            if ((new_lam_N - lam_N)**2 + (new_mu_N - mu_N)**2)**0.5 < self.threshold:
                lam_N = new_lam_N 
                mu_N = new_mu_N
                break 
        
            lam_N = new_lam_N
            mu_N = new_mu_N

        self.mu = mu_N
        self.lamda = lam_N
        self.a = a_N
        self.b = b_N


class ApproximateGaussianMixture():
    """ApproximateGaussianMixture 

    approximately predict posterior distribution for gaussian mixture

    """
    def __init__(self,K,alpha=None,m=None,beta=1,W=None,nu=None,n_iter=1000):
        """

        Args:
            K (int): number of components 
            alpha (1-D array): param of prior distribution of pi 
            m (1-D array): param of prior distribution of mu and Lamda
            beta (float): param of prior distribution of mu and Lamda
            nu (flaot): param of prior distribution of mu and Lamda
            n_iter (int): number of iteration

        """
        self.K = K
        if alpha is None:
            self.alpha = np.ones(self.K) 
        else:
            self.alpha = alpha 
        self.m = m 
        self.beta = beta 
        self.W = W 
        if nu is None:
            self.nu = 1 
        else:
            self.nu = nu
        self.n_iter = n_iter 

    def fit(self,X,initial_responsibility=None,reduce_components=False,threshold=1e-3):
        """

        Args:
            X (2-D array): shape = (N_samples,N_dim), 
            initial_responsibility (2-D array), shape = (N_samples,K) initial respontibility (which should be normalized)
            reduce_components (bool): if E_pi < threshold, that components will be reduced 
            threshold (float): threshold for reducing, this param is used when reduce_components = True

        """

        N = X.shape[0]
        M = X.shape[1]

        if initial_responsibility is None:
            E_z = np.random.rand(N,self.K) + 0.10
            E_z /= E_z.sum(axis = 1,keepdims=True)
        else:
            E_z = initial_responsibility

        if self.m is None:
            self.m = np.zeros(M)
        elif self.m.shape[0] != M:
            raise ValueError("X.shape[1] should be equal to m.shape[0]")

        if self.W is None:
            self.W_inv = np.eye(M)
        else:
            if self.W.shape[0] != M:
                raise ValueError("X.shape[1] should be equal to W.shape[0]")
            self.W_inv = np.linalg.inv(self.W)
        
        for _ in range(self.n_iter):
            
            # M step like
            N_k = E_z.sum(axis = 0)
            x_k_bar = (E_z.T@X) / N_k.reshape(-1,1)
            tmp = X.reshape(N,1,M) - x_k_bar.reshape(1,self.K,M)
            S_k = np.sum(E_z.reshape(N,self.K,1,1)*(tmp.reshape(N,self.K,M,1)*tmp.reshape(N,self.K,1,M)),axis = 0) / N_k.reshape(-1,1,1) 

            alpha = self.alpha + N_k 
            beta = self.beta + N_k 
            m_k = (self.beta*self.m + N_k.reshape(-1,1)*x_k_bar) / beta.reshape(-1,1)
            tmp = x_k_bar - self.m
            W_k = np.linalg.inv(self.W_inv + N_k.reshape(-1,1,1)*S_k + (self.beta*N_k.reshape(-1,1,1)/(self.beta + N_k.reshape(-1,1,1)))*tmp.reshape(-1,M,1)*tmp.reshape(-1,1,M)) 
            nu = self.nu + N_k  

            # E step like
            E_pi = (self.alpha + N_k)/(self.K*self.alpha + N)
            if reduce_components:
                reduce = E_pi >= threshold
                self.alpha = self.alpha[reduce]
                alpha = alpha[reduce]
                beta = beta[reduce]
                m_k = m_k[reduce]
                W_k = W_k[reduce] 
                nu = nu[reduce]
                self.K = reduce.astype("int").sum()

            E_pi = digamma(alpha) - digamma(alpha.sum()) # E_logpi 
            E_Lam = digamma(0.5*(nu + 1 - np.arange(1,M+1,1).reshape(-1,1))).sum(axis = 0) + M*_log(2) + _log(np.linalg.det(W_k))
            tmp = X.reshape(N,1,M) - m_k.reshape(1,self.K,M) 
            E_inner_gauss = M/beta + nu*(tmp.reshape(N,self.K,1,M)@W_k@tmp.reshape(N,self.K,M,1)).reshape(N,self.K)
            
            pho = E_pi + 0.5*E_Lam - M/2*_log(2*np.pi) - 0.5*E_inner_gauss
            E_z = np.exp(pho)
            E_z /= E_z.sum(axis = 1,keepdims=True)
        
        self.r = E_z 
        self.alpha = alpha 
        self.beta = beta 
        self.m_k = m_k 
        self.W_k = W_k 
        self.nu = nu    


        

            






