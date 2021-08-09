"""Sparse Kernel Machines 

    SupportVectorMachineClassifier 
    RelevanceVectorMachineRegressor 
    RelevanceVectorMachineClassifier 

    Todo:
        SupportVecotrMachineRegressor
"""

import numpy as np
from prml.utils.util import sigmoid,kappa
from prml.linear_classifier import Classifier
from prml.kernel_method import BaseKernelMachine


class SupportVectorMachineClassifier(BaseKernelMachine,Classifier):
    """SupportVectorMachineClassifier 

    We use SMO algotithm for solving dual problem

    Attributes:
        C (float) : if C is larger, misclassification on train data is not permitted. 
        eps (float) : parameter for judging KKT condition is fulfilled 
        dual_weight (array): weight 
        b (float): bias parameter 
        support_vector (array): index of support vector 
        support_vector_X (array): explanatory variable of support vector 
        support_vector_y (array): target of support vector 

    Reference:
        https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf

    """
    def __init__(self,C=20.0,eps=1e-3,kernel="Linear",sigma=0.1,a=1.0,b=0.0,h=None,theta=1.0):
        """

        Args: 
            C (float) : if C is larger, misclassification on train data is not permitted. 
            eps (float) : parameter for judging KKT condition is fulfilled 
            kernel (string) : kernel type (default "Linear"). you can choose "Linear","Gaussian","Sigmoid","RBF","Exponential"
            sigma (float) : for "Gaussian" kernel 
            a,b (float) : for "Sigmoid" kernel
            h (function) : for "RBF" kernel 
            theta (float) : for "Exponential" kernel

        """
        super(SupportVectorMachineClassifier,self).__init__(kernel=kernel,sigma=sigma,a=a,b=b,h=h,theta=theta)
        Classifier.__init__(self) # this part should be fixed 
        self.C = C 
        self.eps = eps 
        self.dual_weight = None # alpha
        self.b = None 
        self.support_vector = None 
        self.support_vector_X = None 
        self.support_vector_y = None 

    def fit(self,X,y):
        """fit

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  

        """ 
        self.gram_mat = self.gram_func(X) 
        y = self._onehot_to_label(y)
        y[y == 0] = -1 
        self.y = y 
        self.dual_weight = np.zeros(X.shape[0])
        self.b = 0
        self.error_cache = self._calc_error()
        self.support_vector = np.arange(X.shape[0])[np.logical_and(self.dual_weight > self.eps,self.dual_weight < self.C - self.eps)] 

        num_changed = 0 
        examin_all = True 
        while num_changed > 0 or examin_all:
            num_changed = 0 
            if examin_all:
                for i in range(X.shape[0]):
                    num_changed += self._examin_example(i)
            else:
                for i in self.support_vector:
                    num_changed += self._examin_example(i) 
            if examin_all:
                examin_all = False 
            elif num_changed == 0:
                examin_all = True
            
        self.support_vector_X = X[self.support_vector] 
        self.support_vector_y = y[self.support_vector]
        self.dual_weight = self.dual_weight[self.support_vector] 

        del self.gram_mat,self.y,self.error_cache # for memory
    
    def _examin_example(self,i2):
        """_examin_example 

        Args:   
            i2 (int) : index of the data which will be optimized 
        
        Returns:
            num_changed (int) : parameter was changed or not 

        """
        y2 = self.y[i2] 
        alpha2 = self.dual_weight[i2] 
        E2 = self.error_cache[i2] 
        r2 = E2*y2 # equals ui*yi - 1
        if (r2 < -self.eps and alpha2 < self.C) or (r2 > self.eps and alpha2 > 0): # violate KKT conditions 
            if len(self.support_vector) > 1:
                i1 = np.argmax(np.abs(self.error_cache - E2)) 
                if self._take_step(i1,i2,y2,alpha2,E2):
                    return 1  
            for i1 in self.support_vector:
                if self._take_step(i1,i2,y2,alpha2,E2):
                    return 1  
            for i1 in range(self.y.shape[0]):
                if self._take_step(i1,i2,y2,alpha2,E2):
                    return 1 
        return 0 
    
    def _take_step(self,i1,i2,y2,alpha2,E2):
        """_take_step 

        Args:   
            i1,i2 (int) : index of the data which will be optimized 
            y2 (int) : class of data[i2] 
            alpha2 (float) : weight which corresponds to data[i2]
            E2 (float) : error of data[i2]
        
        Returns:
            updated_or_not (bool): if parameter was updated or not 

        """
        if i1 == i2:
            return False 
        alpha1 = self.dual_weight[i1] 
        y1 = self.y[i1] 
        E1 = self.error_cache[i1]

        L,H = self._calc_bound(y1,y2,alpha1,alpha2)
        if L == H:
            return False  
        
        eta = self.gram_mat[i1,i1] - 2*self.gram_mat[i1,i2] + self.gram_mat[i2,i2]
        a2 = alpha2 + y2*(E1 - E2)/eta
        if a2 < L:
            a2 = L 
        elif a2 > H:
            a2 = H 
        
        if abs(a2 - alpha2) < self.eps*(a2 + alpha2 + self.eps): # change is too small
            return False  
        
        a1 = alpha1 + y1*y2*(alpha2 - a2) 
        self._update_threshold(i1,i2,alpha1,alpha2,a1,a2,y1,y2,E1,E2)
        self.dual_weight[i1] = a1 
        self.dual_weight[i2] = a2 
        self.error_cache = self._calc_error() # more faster update is possible 
        self.support_vector = np.arange(self.y.shape[0])[np.logical_and(self.dual_weight > self.eps,self.dual_weight < self.C - self.eps)] 

        return True 

    def _calc_bound(self,y1,y2,alpha1,alpha2):
        """_calc_bound 

        Args:
            y1,y2 (int): class of data[i1],data[i2] 
            alpha1,alpha2 (float): weight which corresponds to data[i1],data[i2]

        """
        if y1 != y2:
            diff = alpha2 - alpha1
            L = max(0,diff)
            H = min(self.C,self.C + diff)
        else:
            total = alpha1 + alpha2
            L = max(0,total - self.C) 
            H = min(self.C,total) 
        return L,H   

    def _calc_error(self):
        """_calc_error  
        """
        pred = np.dot(self.gram_mat.T,self.dual_weight*self.y) + self.b 
        E = pred - self.y
        return E 
    
    def _update_threshold(self,i1,i2,alpha1,alpha2,a1,a2,y1,y2,E1,E2):
        """_update_threshold

        Args:   
            i1,i2 (int): index of the data which will be optimized 
            alpha1,alpha2 (float): weight which corresponds to data[i1],data[i2]
            a1,a2 (float): updated weight which corresponds to data[i1],data[i2]
            y1,y2 (int): class of data[i1],data[i2] 
            E1,E2 (float) : error of data[i1],data[i2]

        """
        if 0 < a1 < self.C:
            self.b -= E1 + y1*(a1 - alpha1)*self.gram_mat[i1,i1] + y2*(a2 - alpha2)*self.gram_mat[i1,i2] 
        elif 0 < a2 < self.C: 
            self.b -= E2 + y1*(a1 - alpha1)*self.gram_mat[i1,i2] + y2*(a2 - alpha2)*self.gram_mat[i2,i2] 
        else:
            b1 = E1 + y1*(a1 - alpha1)*self.gram_mat[i1,i1] + y2*(a2 - alpha2)*self.gram_mat[i1,i2] 
            b2 = E2 + y1*(a1 - alpha1)*self.gram_mat[i1,i2] + y2*(a2 - alpha2)*self.gram_mat[i2,i2]  
            self.b -= (b1 + b2)/2
        return 
    
    def predict(self,X):
        """predict 

        Args:
            X (2-D arrray) : shape = (N_samples,N_dims)
            return_prob (bool) : if True, return probability 

        Returns:
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 
            or if return_prob == True
            y (1-D array) :  always return probability of belonging to class1 in each record 

        """
        gram_mat = np.zeros((self.support_vector_X.shape[0],X.shape[0]))
        for i in range(self.support_vector_X.shape[0]):
            gram_mat[i] = np.array([self.kernel_func(self.support_vector_X[i],X[j]) for j in range(X.shape[0])]) 
        
        sign_y = np.dot(gram_mat.T,self.dual_weight*self.support_vector_y) + self.b 
        y = np.zeros(X.shape[0])
        y[sign_y > 0] = 1 
        y[sign_y <= 0] = 0 
        return self._inverse_transform(y)
    
    def number_of_support_vector(self):
        """number_of_support_vector

        Returns:
            number_of_support_vector (int) : number of support vector

        """
        return len(self.support_vector) 
    
    def index_of_support_vector(self):
        """index_of_support_vector

        Returns:
            index_of_support_vector (array) : index_of_support_vector

        """
        return self.support_vector


class RelevanceVectorMachineRegressor(BaseKernelMachine):
    """Relevance Vecotr Machine Regressor 

    Attributes:
        alpha,beta (float): hyper parameter 
        max_iter (int): max iteration when model optimize parameters 
        threshold (float) : if error is lower than this, stop iteration
        weight (array) : weight 
        sigma (array): sigma 
        relevance_vector (array): index of relevance vector 
        relevance_vector_X (array): explanatory variable of relevance vector 

    """
    def __init__(self,alpha=None,beta=None,max_iter=100,threshold=1e-7,kernel="Linear",sigma=0.1,a=1.0,b=0.0,h=None,theta=1.0):
        super(RelevanceVectorMachineRegressor,self).__init__(kernel=kernel, sigma=sigma, a=a, b=b, h=h, theta=theta)
        self.alpha = alpha 
        self.beta = beta 
        self.max_iter = max_iter
        self.threshold = threshold
        self.weight = None 
        self.sigma = None 
        self.relevance_vector = None 
        self.relevance_vector_X = None 

    def fit(self,X,y,optimize_param=True):
        """fit 

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,1) 
            optimizer_param (bool): if alpha,beta will be optimized or not

        """

        design_mat = self.gram_func(X) 
        if self.alpha is None:
            self.alpha = np.random.randn(design_mat.shape[1]) 
        if self.beta is None:
            self.beta = np.random.randn(1)[0]

        self.sigma = np.linalg.inv(np.diag(self.alpha) + self.beta*design_mat.T@design_mat) 
        self.weight = self.beta*self.sigma@design_mat.T@y 

        if optimize_param:
            self._optimize(design_mat,X,y)
        else:
            self.relevance_vector = np.arange(X.shape[0]).reshape(-1,1) 
            self.relevance_vector_X = X 

    def _optimize(self,design_mat,X,y):
        """_optimize 

        Args:
            design_mat (2-D array) : design_mat of explanatory variable,shape = (N_samples,N_dim)
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (2-D array) : target variable, shape = (N_samples,1) 

        """
        N = design_mat.shape[0] 

        for _ in range(self.max_iter): 
            gamma = 1 - np.diag(self.sigma)*self.alpha  
            self.alpha = gamma/self.weight.ravel()**2 
            self.alpha = np.clip(self.alpha,0,1e10)
            self.beta = (N - gamma.sum()) / np.sum((y - design_mat@self.weight)**2)
            self.sigma = np.linalg.pinv(np.diag(self.alpha) + self.beta*design_mat.T@design_mat) 
            weight = self.beta*self.sigma@design_mat.T@y 

            if np.mean((weight - self.weight)**2) < self.threshold:
                self.weight = weight
                break 
            self.weight = weight
        
        relevance_vector = np.abs(self.weight) > 1e-3
        self.weight = self.weight[relevance_vector].reshape(-1,1)
        n = self.weight.shape[0]
        self.sigma = self.sigma[np.logical_and(relevance_vector,relevance_vector.ravel())].reshape(n,n)
        self.relevance_vector = np.arange(N)[relevance_vector.ravel()].reshape(-1,1)
        self.relevance_vector_X = X[relevance_vector].reshape(-1,1)
    
    def predict(self,X,return_std=False):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)
            return_std (bool) : if std is returned or not 

        Returns:
            y (2-D array) : predicted value, shape = (N_samples,N_target) 

        """

        design_mat = np.zeros((self.relevance_vector_X.shape[0],X.shape[0]))
        for i in range(self.relevance_vector_X.shape[0]):
            design_mat[i] = np.array([self.kernel_func(self.relevance_vector_X[i],X[j]) for j in range(X.shape[0])]) 
        
        y = design_mat.T@self.weight 
        if return_std:
            std = 1/self.beta + np.diag(design_mat.T@self.sigma@design_mat) 
            return y,std.reshape(-1,1) 
        else:
            return y
    
    def number_of_relevance_vector(self):
        """number_of_relevance_vector

        Returns:
            number_of_relevance_vector (int) : number_of_relevance_vector

        """
        return len(self.relevance_vector) 
    
    def index_of_relevance_vector(self):
        """index_of_relevance_vector

        Returns:
            index_of_relevance_vector (array) : index_of_relevance_vector

        """
        return self.relevance_vector


class RelevanceVectorMachineClassifier(BaseKernelMachine,Classifier):
    """RelevanceVectorMachineClassifier

    Attributes:
        alpha (float): hyper parameter 
        max_iter (int): max iteration when model optimize parameters 
        threshold (float) : if error is lower than this, stop iteration
        weight (array) : weight 
        sigma (array): sigma 
        relevance_vector (array): index of relevance vector 
        relevance_vector_X (array): explanatory variable of relevance vector 

    """
    def __init__(self,alpha=None, max_iter=100,threshold=1e-7,kernel="Linear", sigma=0.1, a=1.0, b=0.0, h=None, theta=1.0):
        super(RelevanceVectorMachineClassifier,self).__init__(kernel=kernel, sigma=sigma, a=a, b=b, h=h, theta=theta)
        Classifier.__init__(self) 
        self.alpha = alpha 
        self.max_iter = max_iter 
        self.threshold = threshold 
        self.weight = None 
        self.sigma = None
        self.relevance_vector = None 
        self.relevance_vector_X = None  

    def fit(self,X,y,optimize_param=True):
        """fit

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dims) 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  
            optimizer_param (bool): if alpha,beta will be optimized or not

        """ 
        y = self._onehot_to_label(y) 
        y = y.reshape(-1,1)

        design_mat = self.gram_func(X) 
        self.weight = np.random.randn(design_mat.shape[1],1) 
        if self.alpha is None:
            self.alpha = np.random.randn(design_mat.shape[1],1) 

        for _ in range(self.max_iter):
            y_pred = sigmoid(design_mat@self.weight) 
            grad = self.alpha*self.weight - design_mat.T@(y - y_pred) 
            B = y_pred*(1 - y_pred)
            H_inv = np.linalg.pinv(design_mat.T@(B*design_mat) + np.diag(self.alpha.ravel()))  
            new_weight = self.weight - H_inv@grad 
            if np.mean((new_weight - self.weight)**2) < self.threshold:
                self.weight = new_weight
                break 
            self.weight = new_weight
            self.sigma = H_inv 

            if optimize_param:
                for _ in range(int(self.max_iter/10)):
                    gamma = 1 - self.alpha*np.diag(H_inv).reshape(-1,1) 
                    new_alpha = gamma/self.weight**2 
                    new_alpha = np.clip(new_alpha,0,1e10)
                    if np.mean((new_alpha - self.alpha)**2) < self.threshold:
                        self.alpha = new_alpha
                        break 
                    self.alpha = new_alpha
            
            else:
                continue
        
        relevance_vector = np.abs(self.weight) > 1e-3
        self.weight = self.weight[relevance_vector].reshape(-1,1)
        n = self.weight.shape[0]
        self.sigma = self.sigma[np.logical_and(relevance_vector,relevance_vector.ravel())].reshape(n,n)
        self.relevance_vector = np.arange(X.shape[0])[relevance_vector.ravel()].reshape(-1,1)
        self.relevance_vector_X = X[relevance_vector.ravel()]

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
        design_mat = np.zeros((self.relevance_vector_X.shape[0],X.shape[0]))
        for i in range(self.relevance_vector_X.shape[0]):
            design_mat[i] = np.array([self.kernel_func(self.relevance_vector_X[i],X[j]) for j in range(X.shape[0])]) 
        
        logit = (design_mat.T@self.weight).ravel() 
        if return_prob:
            sigma = np.diag(design_mat.T@self.sigma@design_mat) 
            prob = sigmoid(kappa(sigma)*logit) 
            return prob  
        else:
            y = np.zeros(X.shape[0])
            y[logit >= 0] = 1
            return self._inverse_transform(y)  
    
    def number_of_relevance_vector(self):
        """number_of_relevance_vector

        Returns:
            number_of_relevance_vector (int) : number_of_relevance_vector

        """
        return len(self.relevance_vector) 
    
    def index_of_relevance_vector(self):
        """index_of_relevance_vector

        Returns:
            index_of_relevance_vector (array) : index_of_relevance_vector

        """
        return self.relevance_vector
            
        
