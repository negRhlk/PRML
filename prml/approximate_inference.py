"""Approximate Inference 

This module is about chapter10. 
VariationalGauss1D, VariationalGaussianMixture, VariationalLogisticRegression are implemented. 

Todo:
    EP_for_noisy_data

"""

import numpy as np 
from math import gamma 
from scipy.special import psi as digamma 

from prml.utils.util import _log,sigmoid,kappa
from prml.linear_classifier import _logistic_regression_base

class VariationalGauss1D():
    """VariationalGauss1D

    approximately predict posterior distribution for gaussian

    Attributes:
        a,b (float): parameter for posterior distibution
        mu,lamda (float): parameter for posterior distibution
        max_iter (int): max iteration 
        threshold (float): threshold 

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


class VariationalGaussianMixture():
    """VariationalGaussianMixture 

    approximately predict posterior distribution for gaussian mixture

    n_sample = N 
    n_components = K 
    n_dim = D (expressed as M in fit())

    Attributes:
        K (int): number of components 
        alpha (1-D array): shape = (K),param of posterior distribution of pi 
        beta (1-D array): shape = (K),param of posterior distribution of mu and Lamda
        m_k (2-D array): shape = (K,D),param of posterior distribution of mu and Lamda
        W_k (3-D array): shape = (K,D,D),param of posterior distribution of mu and Lamda
        nu (1-D array): shape = (K),param of posterior distribution of mu and Lamda
        n_iter (int): number of iteration

        r (2-D array): shape = (N,K),responsibility for data X (after fit() is called)

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

    def _student(self,X,D,mu,L,nu):
        norm_const = gamma(D/2 + nu/2)/gamma(nu/2) * np.linalg.det(L)**0.5/(np.pi*nu)**(D/2) 
        X -= mu
        mahalanobis_dist = (X.reshape(-1,1,D)@L@X.reshape(-1,D,1)).ravel()
        st = (1 + mahalanobis_dist/nu)*(-D/2 - nu/2) 
        return norm_const*st 
    
    def prob_density(self,X):
        """

        predictive density is a mixture of student's t-distributions

        Args:
            X (2-D array): shape = (N_samples,N_dims) 
        
        Returns:
            1-D array: length = N_samples, which is the probability of predictive density

        """
        D = X.shape[1]
        prob = np.zeros(X.shape[0])
        for i in range(self.K):
            L = (self.nu[i] + 1 - D)*self.beta[i]/(1 + self.beta[i])*self.W_k[i]
            prob += self.alpha[i]*self._student(X,D,self.m_k[i],L,self.nu[i]+1-D) 
        prob /= self.alpha.sum()
        return prob 


class VariationalLogisticRegression(_logistic_regression_base):
    """VariationalLogisticRegression

    m_0 = 0 
    S_0 = alpha^-1 * I

    Attributes:
        alpha (float): param for prior distribution for weight
        xi (1-D array): variational param 
        a,b (float): param for posterior density for alpha 
        weight (1-D array): mean of posterior density of weight 
        S (2-D array): std of posterior density of weight
        max_iter (int) : max iteration for parameter optimization
        threshold (float) : threshold for optimizint parameters 

    """
    def __init__(self,alpha=1e-1,max_iter=100,threshold=1e-2,basis_function="gauss",mu=None,s=None,deg=None):
        """

        Args:
            alpha (float): param for prior distribution for weight
            max_iter (int) : max iteration for parameter optimization
            threshold (float) : threshold for optimizint parameters 
            basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features

        """
        super(VariationalLogisticRegression,self).__init__(max_iter,threshold,basis_function,mu,s,deg)
        self.alpha = alpha
        self.xi = None 
        self.a = None 
        self.b = None 

    def fit(self,X,y,init_xi=None,optimize_param=False,init_a=0,init_b=0):
        """fit

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data. 
            init_xi (1-D array): initial param for xi (variational param)
            optimize_param (bool): if alpha will be optimized or not
            init_a,init_b (float): initial param for a,b

        """

        y = self._onehot_to_label(y)

        if init_xi is None:
            xi = np.random.randn(X.shape[0])
        elif init_xi.shape[0] != X.shape[0]:
            raise ValueError("init_xi.shape[0] should be equal to X.shape[0]")
        else:
            xi = init_xi
        
        self.a = init_a
        self.b = init_b

        design_mat = self.make_design_mat(X)
        M = design_mat.shape[1]

        for i in range(self.max_iter):

            # E step
            lamda = self._lamda(xi) 
            self.S = np.linalg.inv(self.alpha*np.eye(M) + 2*(lamda*design_mat.T)@design_mat)
            self.weight = self.S@np.sum((y - 0.5)*design_mat.T,axis = 1,keepdims=True)

            # M step 
            if optimize_param:
                new_xi = self._opt_param(init_a,init_b,design_mat)
            else:
                new_xi = np.diag(design_mat@(self.S + self.weight*self.weight.ravel())@design_mat.T)**0.5 

            if np.mean((xi - new_xi)**2)**0.5 < self.threshold:
                xi = new_xi 
                break 

            xi = new_xi

        self.xi = xi 
    
    def _lamda(self,xi):
        return (sigmoid(xi) - 0.5)/(2*xi)
    
    def _opt_param(self,init_a,init_b,design_mat):
        """

        Args:
            init_a,init_b (float): initial param for a,b
            design_mat (2-D array): design_mat
        
        Returns:
            1-D array: xi 

        """

        M = design_mat.shape[1]
        E_wwT = self.S + self.weight*self.weight.ravel()
        E_wTw = np.diag(E_wwT).sum()

        self.a = init_a + M/2 
        self.b = init_b + E_wTw/2 
        self.alpha = self.a/self.b 
        return np.diag(design_mat@E_wwT@design_mat.T)**0.5

    def predict(self,X,return_prob=False):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
            return_prob (bool) : if True, return probability 

        Returns:
            1-D array or 2-D array: if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting.
                                    If return_prob == True, always return probability of belonging to class1 in each record 

        """
        design_mat = self.make_design_mat(X)
        logit = (design_mat@self.weight).ravel()
        if return_prob:
            sigma = np.diag(design_mat@self.S@design_mat.T) 
            prob = sigmoid(kappa(sigma)*logit)
            return prob 
        else:
            y = np.zeros(X.shape[0])
            y[logit >= 0] = 1 
            return self._inverse_transform(y) 


# if v < 0, what does (2*pi*v)**(D/2) mean? this cause error

# class EP_for_noisy_data():
#     """EP for noisy data

#     Attributes:
#         D (int): data dimension
#         m (1-D array): shape = (D), mean of prior distribution
#         v (float): std of prior distribution 
#         Ss (1-D array): param of each factor 
#         Ms (2-D array): mean of each factor 
#         max_iter (int) : max iteration for parameter optimization
#         threshold (float) : threshold for optimizint parameters 

#     """
#     def __init__(self,max_iter=100,threshold=1e-4):
#         self.max_iter = max_iter 
#         self.threshold = threshold

#     def fit(self,X,a=10,b=100,w=0.5,m=None,v=None):
#         """fit 

#         Args:
#             X (2-D array): shape = (N,D), data
#             a,b,w (float): param of noisy data
#             m (1-D array): shape = (D),initial param for prior distribution
#             v (float): initial param for prior distribution

#         """

#         N = X.shape[0] 
#         D = X.shape[1] 

#         if m is None: 
#             m = np.random.randn(D)
#         elif m.shape[0] != D:
#             raise ValueError("m.shape[0] != X.shape[1]")
        
#         if v is None:
#             v = 1 
        
#         Ss = np.random.randn(N)
#         Ms = np.random.randn(N,D)
#         Vs = np.ones(N)

#         def inv(a):
#             if a > 0:
#                 return 1/(a + 1e-5)
#             else:
#                 return 1/(a - 1e-5)
#         def gauss(x,m,v):
#             return inv(2*np.pi*v)**(D/2)*np.exp(-np.sum((x-m)**2)*inv(v))

#         for _ in range(self.max_iter):
#             param_diff_max = -1e20
#             for i in range(N):
#                 v_n = inv(inv(v) - inv(Vs[i]))
#                 m_n = m + v_n*inv(Vs[i])*(m - Ms[i])
#                 Z_n = (1 - w)*gauss(X[i],m_n,v_n+1) + w*gauss(X[i],0,a) 

#                 pho_n = 1 - w*inv(Z_n)*gauss(X[i],0,a)
#                 new_m = m_n + pho_n*v_n*inv(v_n + 1)*(X[i] - m_n) 
#                 new_v = v_n - pho_n*(v_n)**2*inv(v_n + 1) + pho_n*(1 - pho_n)*(v_n)**2*np.sum((X[i] - m_n)**2)*inv(D*(v_n + 1)**2) 

#                 new_Vi = inv(inv(new_v) - inv(v_n)) 
#                 new_Mi = m_n + (Vs[i] + v_n)*inv(v_n)*(new_m - m_n) 
#                 new_Si = Z_n*inv((2*np.pi*Vs[i])**(D/2)*gauss(Ms[i],m_n,Vs[i]+v_n))

#                 param_diff_max = max(param_diff_max,np.abs(new_m - m).max())
#                 param_diff_max = max(param_diff_max,abs(new_v - v))
#                 param_diff_max = max(param_diff_max,abs(new_Vi - Vs[i])) 
#                 param_diff_max = max(param_diff_max,np.abs(new_Mi - Ms[i]).max())
#                 param_diff_max = max(param_diff_max,abs(new_Si - Ss[i]))

#                 m = new_m 
#                 v = new_v
#                 Vs[i] = new_Vi
#                 Ms[i] = new_Mi
#                 Ss[i] = new_Si

#             if param_diff_max < self.threshold:
#                 break 
        
#         self.D = D
#         self.m = m 
#         self.v = v 
#         self.Ss = Ss 
#         self.Ms = Ms 
#         self.Vs = Vs 
#         return m,v,Ss,Ms,Vs
    
#     def calc_posterior_mean(self):
#         """calc_posterior_mean

#         Returns:
#             mean (1-D array): mean of posterior density for theta

#         """
#         return self.m
    
#     def calc_evidence(self):
#         """calc_evidence

#         Returns:
#             evidence (float): model evidence 

#         """
#         B = np.dot(self.m,self.m)/self.v - np.diag(self.Ms@self.Ms.T)/self.Vs 
#         return (2*np.pi*self.v)**(self.D/2)*np.exp(B/2)*np.prod(self.Ss*(2*np.pi*self.Vs)**(-self.D/2))