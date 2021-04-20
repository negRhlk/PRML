"""Linear Regression 

Chapter3 

- LinearRegression
- Ridge
- BayesianLinearRegression 

"""

import numpy as np 
import matplotlib.pyplot as plt 
import warnings 

from prml.design_mat import GaussMat,SigmoidMat,PolynomialMat

class Regression():
    def __init__(self,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            basis_funtion (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        if basis_function == "gauss":
            self.make_design_mat = GaussMat(mu = mu,s = s)
        elif basis_function == "sigmoid":
            self.make_design_mat = SigmoidMat(mu = mu,s = s)
        elif basis_function == "polynomial":
            self.make_design_mat = PolynomialMat(deg = deg) 
        
        self.weight = None


class LinearRegression(Regression):
    """Linear regression 
    
    Attributes:
        beta (float) : precision parameter 
    """
    def __init__(self,basis_function="gauss",mu=None,s=None,deg=None): 
        """
        Args:
            basis_funtion (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        super(LinearRegression,self).__init__(basis_function,mu,s,deg) 
        self.beta = None 
    
    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,N_target) 
        """ 
        N = X.shape[0] 
        K = y.shape[1] 
        design_mat = self.make_design_mat(X) 
        self.weight = np.linalg.inv(design_mat.T@design_mat)@design_mat.T@y  
        tmp = y - self.weight.T@design_mat.T 
        self.beta = N*K/np.sum(tmp**2) 
    
    def predict(self,X):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)
        Returns:
            y (2-D array) : predicted value, shape = (N_samples,N_target) 
        """ 
        design_mat = self.make_design_mat(X) 
        return np.dot(design_mat,self.weight)


class Ridge(Regression):
    """Ridge

    Attributes:
        lamda (float) : regularization parameter 
    """
    def __init__(self,lamda=1e-2,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            lamda (float) : regularization parameter 
            basis_funtion (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        super(Ridge,self).__init__(basis_function,mu,s,deg) 
        self.lamda = lamda 
    
    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,N_target) 
        """ 
        design_mat = self.make_design_mat(X) 
        M = design_mat.shape[1] 
        self.weight = np.linalg.inv(self.lamda*np.eye(M) + design_mat.T@design_mat)@design_mat.T@y 
    
    def predict(self,X):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)
        Returns:
            y (2-D array) : predicted value, shape = (N_samples,N_target) 
        """ 
        design_mat = self.make_design_mat(X)
        return np.dot(design_mat,self.weight)

class BayesianLinearRegression(Regression):
    """Bayesian linear regression 

    """
    def __init__(self,alpha=1e-1,beta=1e-1,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            alpha (float) : regularization parameter 
            beta (float) : precision parameter 
            basis_funtion (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        
        Node : 
            alpha/beta performs as regularization parameter 
        """
        super(BayesianLinearRegression,self).__init__(basis_function,mu,s,deg) 
        self.S = None 
        self.M = None 
        self.N = 0 
        self.alpha = alpha 
        self.beta = beta 
        
    def fit(self,X,y,optimize_evidence=False,max_iter=20,threshold=1e-3):
        """fit 

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (1-D array) : target variable, shape = (N_samples) 
            optimize_evidence (bool) : if True, alpha and beta are optimized 
            max_iter (int) : number of max iteration for optimizing parameter 
            threshold (float) : if error is lower than this, stop iteration
        """

        self.N = X.shape[0]
        
        if optimize_evidence:
            self._optimize_evidence(X,y,max_iter,threshold) 
            
        design_mat = self.make_design_mat(X) 
        self.M = design_mat.shape[1] 
        self.S = np.linalg.inv(self.alpha*np.eye(self.M) + self.beta*design_mat.T@design_mat) 
        self.weight = self.beta*self.S@design_mat.T@y 
    
    def partial_fit(self,X,y):
        """partial fit 
        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (1-D array) : target variable, shape = (N_samples) 
        Note:
            before this method is called, fit() shouled be called
        """

        self.N += X.shape[0] 
        design_mat = self.make_design_mat(X) 
        S_old_inv = np.linalg.inv(self.S) 
        
        self.S = np.linalg.inv(S_old_inv + self.beta*design_mat.T@design_mat)
        self.weight = self.S@(S_old_inv@self.weight + self.beta*design_mat.T@y)
        
    def _calc_evidence(self,tmp):
        """caluculate evidence 
        """
        E = self.beta/2*tmp + self.alpha/2*np.dot(self.weight,self.weight) 
        evidence = self.M*np.log(self.alpha)/2 + self.N*np.log(self.beta)/2 - E + np.linalg.det(self.S) - self.N*np.log(2*np.pi)/2 
        return evidence 
    
    def _optimize_evidence(self,X,y,max_iter,threshold):
        """optimize evidence
        Args:
            max_iter (int) : number of max iteration for optimizing parameter 
            threshold (float) : if error is lower than this, stop iteration
        """

        design_mat = self.make_design_mat(X) 
        self.M = design_mat.shape[1]
        C = design_mat.T@design_mat
        org_lambdas,_ = np.linalg.eig(C)
        with warnings.catch_warnings(): # Ignore Warnings
            warnings.simplefilter('ignore')
            org_lambdas = org_lambdas.astype(np.float64) 
        before_evidence = -10**10 
        
        for _ in range(max_iter):
            self.S = np.linalg.inv(self.alpha*np.eye(self.M) + self.beta*C) 
            self.weight = self.beta*self.S@design_mat.T@y 
            
            lambdas = self.beta*org_lambdas
            gamma = np.sum(lambdas/(lambdas + self.alpha))
            self.alpha = gamma/np.dot(self.weight,self.weight)
            tmp = y - design_mat@self.weight
            tmp = np.dot(tmp,tmp)
            self.beta = (self.N - gamma)/tmp 
            evidence = self._calc_evidence(tmp)
            
            if np.abs(before_evidence-evidence) < threshold:
                break
            before_evidence = evidence
            
    def predict(self,X,return_std = False):
        """predict

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            return_std (bool) : if True, also return std 
        
        Returns:
            y (1-D array) : predicted value,shape = (N_samples)
            std (1-D array) : std of predicted value, shape = (N_samples)
        """
        design_mat = self.make_design_mat(X) 
        pred = np.dot(design_mat,self.weight).ravel()
        if return_std:
            std = np.sqrt(1/self.beta + np.diag(design_mat@self.S@design_mat.T))
            return pred,std
        else:
            return pred    