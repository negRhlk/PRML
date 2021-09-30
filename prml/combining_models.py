"""combining models 

chapter14
AdaBoost
CARTRegressor 
CARTClassifier 
LinearMixture 
LogisticMixture 

Todo:
    In LogisticMixture, inverse matrix cannot be calculated

"""

import numpy as np 
from abc import ABC,abstractclassmethod

from numpy.core.fromnumeric import sort

from prml.utils.util import _log,sigmoid
from prml.linear_classifier import Classifier,_logistic_regression_base
from prml.linear_regression import Regression

class AdaBoost(Classifier):
    """AdaBoost 

    weak_learner is decision stump

    Attributes:
        M (int): number of weak leaner 
        weak_leaner (list): list of data about weak learner 

    """
    def __init__(self,M=5) -> None:
        """__init__

        Args:
            M (int): number of weak leaner 

        """
        super(AdaBoost,self).__init__()
        self.M = M 

    def fit(self,X,y):
        """fit 

        only accept N_dim = 2 data

        Args:
            X (2-D array): shape = (N_samples,2),
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  

        """
        y = self._onehot_to_label(y) 
        y[y == 0.0] = -1.0
        y = y.astype("int")
        N = len(X)
        sort_idx = np.argsort(X,axis=0)
        weight = np.ones(N)/N 
        weak_learner = [None]*self.M 

        for i in range(self.M):
            x_border,x_more_or_less,x_score = self._weak_learn(X[:,0],sort_idx[:,0],y,weight) 
            y_border,y_more_or_less,y_score = self._weak_learn(X[:,1],sort_idx[:,1],y,weight) 

            if x_score < y_score:
                ax = "x"
                border,more_or_less = x_border,x_more_or_less
            else:
                ax = "y" 
                border,more_or_less = y_border,y_more_or_less
            
            miss = self._miss_idx(X,y,ax,border,more_or_less) 
            eps = np.sum(miss*weight)/np.sum(weight) 
            alpha = _log((1 - eps)/eps) 
            weight *= np.exp(alpha*miss) 

            weak_learner[i] = {
                "ax":ax,
                "border":border,
                "more_or_less":more_or_less,
                "alpha":alpha 
            }

        self.weak_learner = weak_learner 

    def _weak_learn(self,X,sort_idx,y,weight):
        weight_sum = weight.sum()
        more_score = weight[y != 1].sum() # score when all data is asigned 1

        border,more_or_less,score = X[sort_idx[0]]-1,"more",more_score 
        for i in range(len(X)):
            if y[sort_idx[i]] == 1:
                more_score += weight[sort_idx[i]]
            else:
                more_score -= weight[sort_idx[i]]

            less_score = weight_sum - more_score
            if more_score < score:
                border,more_or_less,score = X[sort_idx[i]],"more",more_score
            if less_score < score:
                border,more_or_less,score = X[sort_idx[i]],"less",less_score
        
        return border,more_or_less,score 
    
    def _miss_idx(self,X,y,ax,border,more_or_less):
        y_pred = self._predict(X,ax,border,more_or_less) 
        return (y_pred != y).astype("int")

    def _predict(self,X,ax,border,more_or_less):
        if more_or_less == "more":
            if ax == "x":
                class1 = X[:,0] > border 
            elif ax == "y":
                class1 = X[:,1] > border 
        elif more_or_less == "less":
            if ax == "x":
                class1 = X[:,0] <= border 
            elif ax == 'y':
                class1 = X[:,1] <= border 
        pred = np.zeros(len(X)) - 1
        pred[class1] = 1 
        return pred 

    def predict(self,X):
        """predict 

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,2)
        
        Returns: 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 

        """

        y_pred = np.zeros(len(X))
        for i in range(self.M):
            pred = self._predict(
                X,
                self.weak_learner[i]["ax"],
                self.weak_learner[i]["border"],
                self.weak_learner[i]["more_or_less"],
            )
            y_pred += self.weak_learner[i]["alpha"]*pred
        
        y_pred = np.sign(y_pred)
        return self._inverse_transform(y_pred)


class CARTRegressor():
    """CARTRegressor 

    Attributes:
        lamda (float): regularizatioin parameter
        tree (object): parameter 

    """
    def __init__(self,lamda=1e-2):
        """__init__
        
        Args:
            lamda (float): regularizatioin parameter

        """
        self.lamda = lamda 

    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (1-D array) : target variable, shape = (N_samples) 
            
        """

        N = len(X)
        leaves = np.zeros(N)
        num_nodes = 1
        num_leaves = 1
        tree = []

        while True:
            if num_leaves == 0:
                break
            for leaf in range(num_nodes-num_leaves,num_nodes):
                idx = np.arange(N)[leaf == leaves]
                if len(idx) == 1:
                    num_leaves -= 1
                    tree.append({
                        "border": None, 
                        "target": y[idx][0]
                    }) # has no child
                    continue

                ax,border,score,more_index,less_index = -1,None,1e20,None,None
                for m in range(X.shape[1]):
                    now_border,now_score,now_more_index,now_less_index = self._find_boundry(idx,X[idx,m],y[idx])
                    if now_score < score:
                        ax,border,score,more_index,less_index = m,now_border,now_score,now_more_index,now_less_index

                if border is None: 
                    num_leaves -= 1
                    tree.append({
                        "border": None,
                        "target": y[idx].mean()
                    }) # has no child
                    continue

                tree.append({
                    "left_index": num_nodes, 
                    "right_index": num_nodes+1, 
                    "border": border, 
                    "ax": ax
                })

                leaves[less_index] = num_nodes
                leaves[more_index] = num_nodes+1 

                num_nodes += 2 
                num_leaves += 1

        self.tree = tree 

    def _find_boundry(self,idx,X,y):
        n = len(idx)
        sort_idx = np.argsort(X)
        all_sum = np.sum(y)
        right_sum = all_sum 

        # when all data is in one leaf
        score_now = self._error_function(y,right_sum/n) + self.lamda 
        border_index,score = None,score_now
        pred = np.zeros(n)

        for i in range(n-1):
            right_sum -= y[sort_idx[i]]
            left_sum = all_sum - right_sum
            pred[sort_idx[i+1:]] = right_sum/(n-i-1) 
            pred[sort_idx[:i+1]] = left_sum/(i+1)
            score_now = self._error_function(y,pred) + self.lamda*2
            if score_now < score:
                border_index,score = i,score_now
        
        if border_index is None: # no division
            return None,1e20,None,None 

        border = X[sort_idx[border_index]]
        more_index = idx[sort_idx[border_index+1:]] 
        less_index = idx[sort_idx[:border_index+1]]
        return border,score,more_index,less_index 
    
    def _error_function(self,y,pred):
        return np.mean((y-pred)**2)
    
    def _predict(self,X,p_id=0):
        if self.tree[p_id]["border"] is None: 
            return np.zeros(len(X)) + self.tree[p_id]["target"] 
        
        ax = self.tree[p_id]["ax"]
        border = self.tree[p_id]["border"]
        y = np.zeros(len(X))
        y[X[:,ax] > border] = self._predict(X[X[:,ax] > border],p_id=self.tree[p_id]["right_index"])
        y[X[:,ax] <= border] = self._predict(X[X[:,ax] <= border],p_id=self.tree[p_id]["left_index"])
        return y 

    def predict(self,X):
        """predict 

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,N_dim)
        
        Returns: 
            y (1-D array) : predictive value

        """
        y = self._predict(X)
        return y 


class CARTClassifier(Classifier):
    """CARTClassifier 

    Attributes:
        lamda (float): reguralization parameter 
        error_function (str): "gini" or "error_rate" or "cross_entropy"

    """
    def __init__(self,lamda=1e-2,error_function="gini"):
        """__init__

        Arg:
            lamda (float): reguralization parameter 
            error_function (str): "gini" or "error_rate" or "cross_entropy"

        """
        super(CARTClassifier,self).__init__()
        self.lamda = lamda 
        self.error_function = error_function 

    def fit(self,X,y):
        """fit

        Args:
            X (2-D array): shape = (N_samples,2),
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  

        """

        N = len(X)
        y = self._onehot_to_label(y).ravel()
        leaves = np.zeros(N)
        num_nodes = 1
        num_leaves = 1
        tree = []

        while True:
            if num_leaves == 0:
                break
            for leaf in range(num_nodes-num_leaves,num_nodes):
                idx = np.arange(N)[leaf == leaves]
                if len(np.unique(y[idx])) == 1:
                    num_leaves -= 1
                    tree.append({
                        "border": None, 
                        "target": y[idx][0]
                    }) # has no child
                    continue

                ax,border,score,more_index,less_index = -1,None,1e20,None,None
                for m in range(X.shape[1]):
                    now_border,now_score,now_more_index,now_less_index = self._find_boundry(idx,X[idx,m],y[idx])
                    if now_score < score:
                        ax,border,score,more_index,less_index = m,now_border,now_score,now_more_index,now_less_index

                if border is None: 
                    num_leaves -= 1
                    tree.append({
                        "border": None,
                        "target": round(y[idx].mean())
                    }) # has no child
                    continue

                tree.append({
                    "left_index": num_nodes, 
                    "right_index": num_nodes+1, 
                    "border": border, 
                    "ax": ax
                })

                leaves[less_index] = num_nodes
                leaves[more_index] = num_nodes+1 

                num_nodes += 2 
                num_leaves += 1

        self.tree = tree 

    def _find_boundry(self,idx,X,y):
        n = len(idx)
        sort_idx = np.argsort(X)
        all_sum = np.sum(y)
        right_sum = all_sum 

        # when all data is in one leaf,
        # score_now = self._error_function(y,round(right_sum/n)) + self.lamda 
        # border_index,score = None,score_now
        border_index,score = None,1e20
        pred = np.zeros(n)

        for i in range(n-1):
            right_sum -= y[sort_idx[i]]
            left_sum = all_sum - right_sum
            pred[sort_idx[i+1:]] = round(right_sum/(n-i-1))  
            pred[sort_idx[:i+1]] = round(left_sum/(i+1))
            score_now = self._error_function(y,pred) + self.lamda*2
            if score_now < score:
                border_index,score = i,score_now
        
        if border_index is None: # no division
            return None,1e20,None,None 

        border = X[sort_idx[border_index]]
        more_index = idx[sort_idx[border_index+1:]] 
        less_index = idx[sort_idx[:border_index+1]]
        return border,score,more_index,less_index 
    
    def _error_function(self,y,pred):
        if self.error_function == "error_rate":
            return (y != pred).astype("int").sum()/len(y) 

        elif self.error_function == "gini":
            u = np.unique(pred)
            err = 0 
            for cl in u:
                _,count = np.unique(y[pred == cl],return_counts=True)
                class_rate = count/count.sum()
                err += np.sum(class_rate*(1 - class_rate))
            return err 

        elif self.error_function == "cross_entropy":
            u = np.unique(pred)
            err = 0 
            for cl in u:
                _,count = np.unique(y[pred == cl],return_counts=True)
                class_rate = count/count.sum()
                err -= np.sum(class_rate*np.log(class_rate))
            return err 

        else:
            raise ValueError(f"there is no error function whose name is {self.error_function}")
    
    def _predict(self,X,p_id=0):
        if self.tree[p_id]["border"] is None: 
            return np.zeros(len(X)) + self.tree[p_id]["target"] 
        
        ax = self.tree[p_id]["ax"]
        border = self.tree[p_id]["border"]
        y = np.zeros(len(X))
        y[X[:,ax] > border] = self._predict(X[X[:,ax] > border],p_id=self.tree[p_id]["right_index"])
        y[X[:,ax] <= border] = self._predict(X[X[:,ax] <= border],p_id=self.tree[p_id]["left_index"])
        return y 

    def predict(self,X):
        """predict 

        Args:
            X (2-D array) : explanatory variable, shape = (N_samples,2)
        
        Returns: 
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. This depends on parameter y when fitting. 

        """
        y = self._predict(X)
        return self._inverse_transform(y)


class LinearMixture(Regression):
    """LinearMixture

    Attributes:
        K (int): number of mixture modesl 
        max_iter (int): max iteration 
        threshold (float): threshold for EM algorithm 
        pi (1-D array): mixture, which model is chosen
        weight (2-D array): shape = (K,M), M is dimension of feature space, weight
        beta (float): precision parameter

    """
    def __init__(self,K=3,max_iter=100,threshold=1e-3,basis_function="gauss",mu=None,s=None,deg=None):
        super(LinearMixture,self).__init__(basis_function,mu,s,deg)
        self.K = K 
        self.max_iter = max_iter 
        self.threshold = threshold

    def _gauss(self,x,mu,beta):
        return (beta/2*np.pi)**0.5 * np.exp(-beta/2*(x-mu)**2)

    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array) : explanatory variable,shape = (N_samples,N_dim)
            y (1-D array) : target variable, shape = (N_samples) 

        """ 

        design_mat = self.make_design_mat(X)
        N,M = design_mat.shape
        gamma = np.random.rand(N,self.K) + 1
        gamma /= gamma.sum(axis=1,keepdims=True)

        for _ in range(self.max_iter):
             
            # M step 
            pi = gamma.mean(axis = 0)
            R = gamma.T.reshape(self.K,N,1)
            weight = np.linalg.inv(design_mat.T@(R*design_mat))@design_mat.T@(R*y.reshape(-1,1))
            weight = weight.reshape((self.K,M))
            beta = N/np.sum(gamma*(y.reshape(-1,1) - design_mat@weight.T)**2)

            # E step 
            gauss = pi*np.exp(-beta/2*(y.reshape(-1,1) - design_mat@weight.T)**2) + 1e-10
            new_gamma = gauss/gauss.sum(axis=1,keepdims=True) 

            if np.mean((new_gamma - gamma)**2)**0.5 < self.threshold:
                gamma = new_gamma 
                break 

            gamma = new_gamma 
        
        self.pi = pi 
        self.weight = weight 
        self.beta = beta 

    def predict(self,X):
        """predict

        Args:
            X (2-D array) : data,shape = (N_samples,N_dim)
        Returns:
            y (1-D array) : predicted value, shape = (N_samples)

        """ 
        
        design_mat = self.make_design_mat(X)
        return np.dot(design_mat@self.weight.T,self.pi)


class LogisticMixture(_logistic_regression_base):
    """LogisticMixture

    Attributes:
        K (int): number of mixture models
        max_iter (int) : max iteration for parameter optimization
        threshold (float) : threshold for optimizint parameters 
        mu (1-D array) : mean parameter 
        s (1-D array) : standard deviation parameter 
        deg (int) : max degree of polynomial features
    
    Note:
        In many cases, inverse matrix cannot be calculated.

    """
    def __init__(self,K=3,max_iter=100,threshold=1e-3,basis_function="gauss",mu=None,s=None,deg=None):
        """__init__

        Args:
            K (int): number of mixture models
            max_iter (int) : max iteration for parameter optimization
            threshold (float) : threshold for optimizint parameters 
            mu (1-D array) : mean parameter 
            s (1-D array) : standard deviation parameter 
            deg (int) : max degree of polynomial features
            
        """
        super(LogisticMixture,self).__init__(max_iter,threshold,basis_function,mu,s,deg) 
        self.K = K 

    def fit(self,X,y):
        """fit 

        Args:
            X (2-D array): shape = (N_samples,N_dim),
            y (1-D array or 2-D array) : if 1-D array, y should be label-encoded, but 2-D arrray, y should be one-hot-encoded. should be 2-class data.  

        """
        t = self._onehot_to_label(y)
        t = t.reshape(-1,1)
        design_mat = self.make_design_mat(X)
        N = design_mat.shape[0]
        gamma = np.random.rand(N,self.K) + 1
        gamma /= gamma.sum(axis=1,keepdims=True)
        weight = None 

        for _ in range(self.max_iter):
             
            # M step 
            pi = gamma.mean(axis = 0)
            weight = self._Mstep(design_mat,t,gamma,weight) 
            y = sigmoid(design_mat@weight.T) 

            # E step 
            prob = pi*(y**t)*((1-y)**(1-t)) + 1e-10
            new_gamma = prob/prob.sum(axis=1,keepdims=True)

            if np.mean((new_gamma - gamma)**2)**0.5 < self.threshold:
                gamma = new_gamma 
                break 

            gamma = new_gamma 
        
        self.pi = pi 
        self.weight = weight 
    
    def _Mstep(self,design_mat,t,gamma,weight):
        M = design_mat.shape[1] 
        if weight is None:
            weight = np.random.randn(self.K,M) 
        
        for _ in range(4):
            y = sigmoid(design_mat@weight.T) 

            # gradient
            tmp = gamma*(y - t)
            dQ = tmp.T@design_mat 

            # hessian 
            tmp1 = gamma*y*(1 - y) 
            tmp2 = design_mat.reshape(-1,M,1)*design_mat.reshape(-1,1,M) 
            H = np.sum(tmp1.T.reshape(self.K,-1,1,1)*tmp2,axis=1)
            print(H)

            dweight = np.squeeze(np.linalg.inv(H)@dQ.reshape(self.K,M,1))  
            if np.mean(np.abs(dweight)) < self.threshold:
                weight -= dweight
                return weight 
            weight -= dweight
        return weight 

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
        y = np.dot(sigmoid(design_mat@self.weight.T),self.pi)
        if return_prob:
            return y 
        else:
            y = np.zeros(X.shape[0])
            y[y >= 0.5] = 1
            return self._inverse_transform(y)