"""combining models 

    AdaBoost

"""

import numpy as np 
from abc import ABC,abstractclassmethod

from prml.utils.util import _log 
from prml.linear_classifier import Classifier

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