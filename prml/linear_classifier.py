"""Linear Classification Models 

chapter4 
    linear classifier (loss is least squared error) 
    ficher's linear discriminant (for 2class)
    ficher's linear discriminant (for multiple-class)
    perceptron
    generative classifier 

"""

import numpy as np 
from math import log 
from prml.utils.util import softmax
from prml.utils.encoder import OnehotToLabel,LabelToOnehot

class Classifier(): 
    """
    Attributes:
        weight (array) : paremeter
        transformer (object) : transform class encoding
    """
    def __init__(self):
        self.weight = None 
        self.transformer = None
    
    def _onehot_to_label(self,X):
        if X.ndim == 2:
            self.transformer = OnehotToLabel() 
            X = self.transformer.fit_transform(X)
        return X 

    def _label_to_onehot(self,X):
        if X.ndim == 1:
            self.transformer = LabelToOnehot() 
            X = self.transformer.fit_transform(X) 
        return X 
    
    def _inverse_transform(self,X):
        if self.transformer is None:
            return X
        else:
            return self.transformer.inverse(X) 

class LinearClassifier(Classifier):
    """Linear Classifier 

    solve classification problem by minimizing least-squared-error. 
    this is not so good for classification problem. 
    """

    def __init__(self):
        super(LinearClassifier,self).__init__()
    
    def fit(self,X,y):
        """fit

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded 
        """
        y = self._label_to_onehot(y) 
        one = np.ones((X.shape[0],1))
        X = np.hstack((one,X))
        self.weight = np.linalg.inv(X.T@X)@X.T@y 
    
    def predict(self,X):
        """predict 

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims)
        
        Returns: 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 
        """
        one = np.ones((X.shape[0],1))
        X = np.hstack((one,X))
        y = X@self.weight
        label = y.argmax(axis = 1)
        return self._inverse_transform(label) 


class Fisher1D(Classifier):
    """fisher's linear discriminant (for 2-class data)

    project onto the line 
    """
    def __init__(self):
        super(Fisher1D,self).__init__()
        
    def fit(self,X,y):
        """fit

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  
        """
        y = self._onehot_to_label(y) 
        class0 = X[y == 0]
        class1 = X[y == 1]
        m1 = class0.mean(axis = 0)
        m2 = class1.mean(axis = 0)
        S_w = (class0 - m1).T@(class0 - m1) + (class1 - m2).T@(class1 - m2)
        self.weight = np.linalg.inv(S_w)@(m2 - m1).reshape(-1,1) 
        
    def transform(self,X):
        """transform 

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
        
        Returns:
            y (1-D array) : projection to 1D data

        """
        return (X@self.weight).ravel()
    
    def fit_transform(self,X,y):
        """fit and transform
        """
        self.fit(X,y)
        return self.transform(X)


class Fisher(Classifier):
    """fisher's linear discriminant 

    Attributes:
        n_components (int) : the size of dimension in which you want to project the data
    """
    def __init__(self,n_components=None):
        super(Fisher,self).__init__() 
        self.n_components = n_components 
        
    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dim)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded.  
        """
        y = self._onehot_to_label(y)
        K = len(np.unique(y)) 
        D = X.shape[1]
        m = X.mean(axis = 0) 
        S_W = np.zeros((D,D))
        S_B = np.zeros((D,D))
        for k in range(K):
            X_k = X[y == k] 
            m_k = X_k.mean(axis = 0)
            S_W += (X_k - m_k).T@(X_k - m_k)
            S_B += X_k.shape[0]*(m_k - m)@(m_k - m).T
        _,eigen_vec = np.linalg.eig(np.linalg.inv(S_W)@S_B)
        if self.n_components is not None:
            self.weight = eigen_vec[:,:self.n_components]
        else:
            self.weight = eigen_vec[:,:K]  
        
    def transform(self,X):
        """transform 

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
        
        Returns:
            y (array) : projected data, shape = (N_samples,n_components)
        """
        return X@self.weight
    
    def fit_transform(self,X,y):
        """fit and transform 
        """
        self.fit(X,y)
        return self.transform(X)


class Perceptron(Classifier):
    """Perceptron 

    Attributes:
        learning_rate (float) : learning rate 
        max_iter (int) : max iteration
        phi (object) : basis function ([1] + x)  
    """
    def __init__(self,learning_rate=1e-3,max_iter=100):
        """
        Args:
            learning_rate (float) : learning rate 
            max_iter (int) : max iteration
        """
        super(Perceptron,self).__init__() 
        self.learning_rate = learning_rate 
        self.max_iter = max_iter
        self.phi = lambda x: np.concatenate(([1],x))
        
    def fit(self,X,y):
        """fit 
        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data. 
        """
        y = self._onehot_to_label(y) 
        y[y == 0] = -1

        M = X.shape[1] + 1
        self.weight = np.random.randn(M,1) 
        for _ in range(self.max_iter):
            y_pred = self._predict(X)
            incorrect = X[y_pred*y < 0]
            if incorrect.shape[0] == 0:
                return
            slope = np.vstack([self.phi(x)*y[i] for i,x in enumerate(incorrect)]).sum(axis = 0) 
            self.weight += self.learning_rate*slope.reshape(-1,1)
    
    def _predict(self,X):
        """predict while training 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
        Returns:
            y_pred (1-D arrat) : shape = (N_samples) values is 1 or -1
        """
        
        design_mat = np.vstack([self.phi(x) for x in X])
        return np.where(design_mat@self.weight > 0,1,-1).ravel()
        
    def predict(self,X):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 
        """
        design_mat = np.vstack([self.phi(x) for x in X])
        pred = (design_mat@self.weight).ravel()
        y = np.zeros(X.shape[0])
        y[pred > 0] = 1
        return self._inverse_transform(y) 


class GenerativeClassifier(Classifier):
    """Generative Classifier 

    for 2-class 
    this model uses the solution obtained by maximum likehood method. 

    Attributes:
        pi (float) : probability of belongin to class1 
        mu1 (1-D array) : mean of normal distribution of class1 
        mu2 (1-D array) : mean of normal distribution of class2 
        sigma (2-D array) : std of of normal distribution of class1 and class2
        b (float) : bias parameter
    """
    def __init__(self):
        super(GenerativeClassifier,self).__init__() 
        self.pi = 0
        self.mu1 = None 
        self.mu2 = None 
        self.sigma = None 
        self.b = 0 
    
    def fit(self,X,y):
        """fit 
        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data. 
        """
        y = self._onehot_to_label(y)
        
        N1 = y.sum() 
        N2 = y.shape[0] - N1
        N = N1 + N2
        pi = N1/(N1 + N2) 
        
        X1 = X[y == 1]
        X2 = X[y == 0]
        mu1 = X1.sum(axis = 0)/N1 
        mu2 = X2.sum(axis = 0)/N2 
        
        S1 = (X1 - mu1).T@(X1 - mu1)/N1 
        S2 = (X2 - mu2).T@(X2 - mu2)/N2 
        sigma = N1/N*S1 + N2/N*S2    
        
        Sinv = np.linalg.inv(sigma)
        self.weight = np.dot(Sinv,mu1 - mu2) 
        self.b = -mu1.T@np.dot(Sinv,mu1)/2 + mu2.T@np.dot(Sinv,mu2)/2 + log(pi/(1.0 - pi))
        
        self.pi = pi 
        self.mu1 = mu1 
        self.mu2 = mu2 
        self.sigma = sigma 
        
    def predict(self,X):
        """predict 

        Args:
            X (2-D array) : shape = (N_samples,N_dims)
        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 
        """
        
        logit = X@self.weight + self.b 
        prob = softmax(logit)
        y = np.zeros(X.shape[0])
        y[prob > 0.5] = 1
        return self._inverse_transform(y) 
    
    def generate(self,size = 20):
        """generate data 

        Args:
            size (int) : number of data to be generated 
        Returns:
            X (2-D array) : generated data, shape = (N_samples,N_dims)
            y (2-D array) : generated data, onehotencoded, shape = (N_samples,2)
        """
        x = np.random.rand(size) 
        
        X = np.zeros((size,len(self.weight)))
        y = np.zeros((size,2)) 
        y[x >= self.pi,0] = 1 
        y[x < self.pi,1] = 1
        
        class1 = (x < self.pi).astype("int").sum()
        X[x >= self.pi,:] = np.random.multivariate_normal(self.mu2,self.sigma,size - class1) 
        X[x < self.pi,:] = np.random.multivariate_normal(self.mu1,self.sigma,class1)
        
        return X,y