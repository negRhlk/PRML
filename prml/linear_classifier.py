"""Linear Classification Models 

chapter4 
    linear classifier (loss is least squared error) 
    ficher's linear discriminant (for 2class)
    ficher's linear discriminant (for multiple-class)
    perceptron
    generative classifier 
    logistic regression (for 2-class)
    logistic regression (for multi-class)
    bayesian logistic regression (for 2-class)
"""

import numpy as np 
from math import log 
from prml.utils.util import sigmoid,softmax,binary_cross_entropy,cross_entropy,kappa
from prml.utils.encoder import OnehotToLabel,LabelToOnehot
from prml.design_mat import GaussMat,SigmoidMat,PolynomialMat

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
        onehot = np.zeros_like(y)
        for k in range(y.shape[1]):
            onehot[label == k,k] = 1 
        return self._inverse_transform(onehot) 


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
        prob = sigmoid(logit) 
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


class _logistic_regression_base(Classifier):
    def __init__(self,max_iter,threshold,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            basis_funtion (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        super(_logistic_regression_base,self).__init__() 
        if basis_function == "gauss":
            self.make_design_mat = GaussMat(mu = mu,s = s)
        elif basis_function == "sigmoid":
            self.make_design_mat = SigmoidMat(mu = mu,s = s)
        elif basis_function == "polynomial":
            self.make_design_mat = PolynomialMat(deg = deg) 

        self.max_iter = max_iter 
        self.threshold = threshold


class LogisticRegression(_logistic_regression_base):
    """Logistic Regression for 2-class 

    use IRLS when optimiztin parameters 

    Attributes:
        weight (array) : parameters
        max_iter (int) : max iteration for parameter optimization
        threshold (float) : threshold for optimizint parameters 
        basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter 
        deg (int) : max degree of polynomial features
    """
    def __init__(self,max_iter=30,threshold=1e-2,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            max_iter (int) : max iteration for parameter optimization
            threshold (float) : threshold for optimizint parameters 
            basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        super(LogisticRegression,self).__init__(max_iter,threshold,basis_function,mu,s,deg)
        
    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data. 
        
        Note:
            optimizing parameters in IRLS
        """
        target = self._onehot_to_label(y)
        target = target.reshape(-1,1)
        
        design_mat = self.make_design_mat(X) 
        self.weight = np.random.randn(design_mat.shape[1]).reshape(-1,1)
        for _ in range(self.max_iter):
            y = sigmoid(design_mat@self.weight)
            if binary_cross_entropy(target,y) < self.threshold:
                break 
            R = y*(1.0 - y)
            if np.any(abs(R)<1e-20): # prevent overflow and error
                R += 1e-10
            z = design_mat@self.weight - (y - target)/R 
            self.weight = np.linalg.pinv(design_mat.T@(R*design_mat))@design_mat.T@(R*z)
        
    def predict(self,X,return_prob=False):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
            return_prob (bool) : if True, return probability 
        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 

            or if return_prob == True

            y (1-D array) :  always return probability of belonging to class1 in each record 
        """
        
        design_mat = self.make_design_mat(X) 
        logit = (design_mat@self.weight).ravel() 
        if return_prob:
            return  sigmoid(logit)
        else:
            y = np.zeros(X.shape[0])
            y[logit >= 0] = 1
            return self._inverse_transform(y)


class MultiClassLogisticRegression(_logistic_regression_base):
    """Logistic Regression for multi-class

    use gradient descent method for parameter optimization 

    Attributes:
        max_iter (int) : max iteration for parameter optimization
        threshold (float) : threshold for optimizint parameters 
        learning_rate (float) : learning_rate
        basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter 
        deg (int) : max degree of polynomial features
    """
    def __init__(self,max_iter=30,threshold=1e-2,learning_rate=1e-2,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            max_iter (int) : max iteration for parameter optimization
            threshold (float) : threshold for optimizint parameters 
            learning_rate (float) : learning_rate
            basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        super(MultiClassLogisticRegression,self).__init__(max_iter,threshold,basis_function,mu,s,deg)
        self.learning_rate = learning_rate
        
    def fit(self,X,y): 
        """fit 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. 
        
        Note:
            optimizing parameters in gradient descent method 
        """
        y = self._label_to_onehot(y) 
        design_mat = self.make_design_mat(X) 
        self.weight = np.random.randn(design_mat.shape[1],y.shape[1])
        
        for _ in range(self.max_iter):
            probability = softmax(design_mat@self.weight)
            loss = cross_entropy(y,probability)
            if loss < self.threshold:
                break 
            self.weight -= self.learning_rate*design_mat.T@(probability - y)   
        
    def predict(self,X,return_prob=False):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
            return_prob (bool) : if True, return probability 
        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 

            or if return_prob == True

            y (2-D array) :  always return probability of belonging to each class in each record 
        """
        
        design_mat = self.make_design_mat(X) 
        logit = design_mat@self.weight 
        if return_prob:
            return softmax(logit)
        else:
            y = softmax(logit)
            return self._inverse_transform(y) 


class BayesianLogisticRegression(_logistic_regression_base):
    """Bayesian logistic regression for 2-class 

    Attributes:
        weight (array) : parameter 
        S (array) : variance of posterior distribution 
        alpha (float) : precision parameter of prior distribution 
        max_iter (int) : max iteration for parameter optimization
        threshold (float) : threshold for optimizint parameters 
        learning_rate (float) : learning rate 
        basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter 
        deg (int) : max degree of polynomial features
    """
    def __init__(self,alpha=1e-2,max_iter=30,threshold=1e-2,learning_rate=1e-2,basis_function="gauss",mu=None,s=None,deg=None):
        """
        Args:
            alpha (float) : precision parameter of prior distribution 
            max_iter (int) : max iteration for parameter optimization
            threshold (float) : threshold for optimizint parameters 
            learning_rate (float) : learning rate 
            basis_function (str) : "gauss" or "sigmoid" or "polynomial" 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
        """
        super(BayesianLogisticRegression,self).__init__(max_iter,threshold,basis_function,mu,s,deg)
        self.alpha = alpha 
        self.learning_rate = learning_rate
        self.S = None
    
    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : data, shape = (N_samples,N_dims)
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data. 
        
        Note:
            optimizing parameters in gradient descent method 
        """
        y = self._onehot_to_label(y) 
        y = y.reshape(-1,1)
            
        design_mat = self.make_design_mat(X) 
        self.weight = np.random.randn(design_mat.shape[1]).reshape(-1,1) 
        
        for _ in range(self.max_iter):
            probability = sigmoid(design_mat@self.weight)
            loss = binary_cross_entropy(y,probability)
            if loss < self.threshold:
                break
            self.weight -= self.learning_rate*(self.alpha*self.weight + design_mat.T@(probability - y))
        
        R = (probability*(1.0 - probability)).ravel()  
        self.S = 1/self.alpha*np.eye(self.weight.shape[1]) + (R*design_mat.T)@design_mat             
    
    def predict(self,X,return_prob=False):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
            return_prob (bool) : if True, return probability 
        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 

            or if return_prob == True

            y (1-D array) :  always return probability of belonging to class1 in each record 
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