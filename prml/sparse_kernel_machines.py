"""Sparse Kernel Machines 

    SupportVectorMachineClassifier 
    
"""

import numpy as np 
from prml.linear_classifier import Classifier
from prml.kernel_method import BaseKernel, BaseKernelMachine, DualRegression



  
class SupportVectorMachineClassifier(BaseKernelMachine,Classifier):
    """SupportVectorMachineClassifier 

    We use SMO algotithm for solving dual problem

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

        del self.gram_mat,self.y,self.error_cache,self.support_vector # for memory
    
    def _examin_example(self,i2):
        """_examin_example 

        Args:   
            i2 (int) : index of the data which will be optimized 
        
        Returns:
            num_changed (bool) : if parameter was changed of not 
        """
        y2 = self.y[i2] 
        alpha2 = self.dual_weight[i2] 
        E2 = self.error_cache[i2] 
        r2 = E2*y2 # equals ui*yi - 1
        if (r2 < -self.eps and alpha2 < self.C) or (r2 > self.eps and alpha2 > 0): # violate KKT conditions 
            if len(self.support_vector) > 1:
                i1 = np.argmax(np.abs(self.error_cache - E2)) 
                if self._take_step(i1,i2,y2,alpha2,E2):
                    return True 
            for i1 in self.support_vector:
                if self._take_step(i1,i2,y2,alpha2,E2):
                    return True 
            for i1 in range(self.y.shape[0]):
                if self._take_step(i1,i2,y2,alpha2,E2):
                    return True 
        return False 
    
    def _take_step(self,i1,i2,y2,alpha2,E2):
        """_take_step 

        Args:   
            i1,i2 (int) : index of the data which will be optimized 
            y2 (int) : class of data[i2] 
            alpha2 (float) : weight which corresponds to data[i2]
            E2 (float) : error of data[i2]
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
            return 0 
        
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
