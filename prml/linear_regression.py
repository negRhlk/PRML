import numpy as np 
import matplotlib.pyplot as plt 
import warnings 


# Gauss 
def gauss(x,mu,s):
    #:params x: 1-D array data 
    #:prams mu,s: \mu and \sigma 
    #:return: \phi(x) 
    phi = np.exp(-(x.reshape(-1,1) - mu)/(2*s**2))
    phi = np.concatenate(([1],phi.ravel()))
    return phi


class LinearRegression():
    def __init__(self,mu = None,s = None):
        self.weight = None
        self.beta = None 
        self.mu = mu
        self.s = s
        self.phi = lambda x:gauss(x,self.mu,self.s)
        
    def fit(self,X,y):
        #:params X: 2-D array (N_samples,N_dims)
        #:params y: 2-D array (N_samples,N_targets)  
        N = X.shape[0] 
        K = y.shape[1] 
        design_mat = np.vstack([self.phi(x) for x in X])
        self.weight = np.linalg.inv(design_mat.T@design_mat)@design_mat.T@y  
        tmp = y - self.weight.T@design_mat.T 
        self.beta = N*K/np.sum(tmp**2) 
        
    def predict(self,X):
        #:params X: 2-D array (N_samples,N_dims) N_dims = len(mu) = len(s) 
        design_mat = np.vstack([self.phi(x) for x in X])
        return np.dot(design_mat,self.weight)


class Ridge():
    def __init__(self,lamda=1e-2,mu = None,s = None):
        self.weight = None
        self.lamda = lamda 
        self.mu = mu
        self.s = s
        self.phi = lambda x:gauss(x,self.mu,self.s)
        
    def fit(self,X,y):
        #:params X: 2-D array (N_samples,N_dims)
        #:params y: 2-D array (N_samples,N_targets)  
        N = X.shape[0] 
        K = y.shape[1]
        M = X.shape[1]*self.mu.shape[0] + 1
        design_mat = np.vstack([self.phi(x) for x in X])
        self.weight = np.linalg.inv(self.lamda*np.eye(M) + design_mat.T@design_mat)@design_mat.T@y 

    def predict(self,X):
        #:params X: 2-D array (N_samples,N_dims) N_dims = len(mu) = len(s) 
        design_mat = np.vstack([self.phi(x) for x in X])
        return np.dot(design_mat,self.weight)


class BayesianLinearRegression():
    def __init__(self,alpha = 1e-1,beta = 1e-1,mu = None,s = None):
        self.weight = None
        self.S = None
        self.M = None
        self.N = 0
        self.alpha = alpha 
        self.beta = beta 
        self.mu = mu
        self.s = s
        self.phi = lambda x:gauss(x,mu,s)
        
    def fit(self,X,y,optimize_evidence = False,n_iters = 20,threshold = 1e-3):
        #:params X: 2-D array (N_samples,N_dims)
        #:params y: 1-D array (N_samples)  
        #:params optimze_evidence: if alpha and beta is optimized or not 
        
        self.N = X.shape[0]
        self.M = X.shape[1]*self.mu.shape[0] + 1
        
        if optimize_evidence:
            self.optimize_evidence_(X,y,n_iters,threshold) 
            
        design_mat = np.vstack([self.phi(x) for x in X])
        self.S = np.linalg.inv(self.alpha*np.eye(self.M) + self.beta*design_mat.T@design_mat) 
        self.weight = self.beta*self.S@design_mat.T@y 
    
    def partial_fit(self,X,y):
        # Before this method is called, fit() should be called 
        
        self.N += X.shape[0] 
        design_mat = np.vstack([self.phi(x) for x in X])
        S_old_inv = np.linalg.inv(self.S) 
        
        self.S = np.linalg.inv(S_old_inv + self.beta*design_mat.T@design_mat)
        self.weight = self.S@(S_old_inv@self.weight + self.beta*design_mat.T@y)
        
    def calc_evidence_(self,tmp):
        E = self.beta/2*tmp + self.alpha/2*np.dot(self.weight,self.weight) 
        evidence = self.M*np.log(self.alpha)/2 + self.N*np.log(self.beta)/2 - E + np.linalg.det(self.S) - self.N*np.log(2*np.pi)/2 
        return evidence 
    
    def optimize_evidence_(self,X,y,n_iters,threshold):
        #:params n_iters: Number of times to optimize alpha and beta 
        #:params threshold: If the difference of evidence is lower than this, 
        
        design_mat = np.vstack([self.phi(x) for x in X])
        C = design_mat.T@design_mat
        org_lambdas,_ = np.linalg.eig(C)
        with warnings.catch_warnings(): # Ignore Warnings
            warnings.simplefilter('ignore')
            org_lambdas = org_lambdas.astype(np.float64) 
        before_evidence = -10**10 
        
        for _ in range(n_iters):
            self.S = np.linalg.inv(self.alpha*np.eye(self.M) + self.beta*C) 
            self.weight = self.beta*self.S@design_mat.T@y 
            
            lambdas = self.beta*org_lambdas
            gamma = np.sum(lambdas/(lambdas + self.alpha))
            self.alpha = gamma/np.dot(self.weight,self.weight)
            tmp = y - design_mat@self.weight
            tmp = np.dot(tmp,tmp)
            self.beta = (self.N - gamma)/tmp 
            evidence = self.calc_evidence_(tmp)
            
            if np.abs(before_evidence-evidence) < threshold:
                break
            before_evidence = evidence
            
    def predict(self,X,return_std = False):
        design_mat = np.vstack([self.phi(x) for x in X])
        pred = np.dot(design_mat,self.weight).ravel()
        if return_std:
            std = np.sqrt(1/self.beta + np.diag(design_mat@self.S@design_mat.T))
            return pred,std
        else:
            return pred    