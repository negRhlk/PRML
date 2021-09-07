"""Datasets

"""

import numpy as np 
import os 

class RegressionDataGenerator():
    """RegressionDataGenerator

    Create 1-D toy data for regression 
    """
    def __init__(self,f):
        """
        Args:
            f (object) : generate 1-D data which follows f(x) + gauss noise 
        """
        self.f = f 
    
    def __call__(self,n = 50,lower = 0,upper = 2*np.pi,std = 1): 
        """Make data 
        Args:
            n (int) : number of data 
            lower,upper (float) : generate data almost lower <= x <= upper 
            std (float) : std of gauss noise
        Returns:
            X (2-D array) : explanatory variable,shape = (N_samples,1)
            y (2-D array) : target variable, shape = (N_samples,1) 
        """ 
        X = np.random.rand(n)*(lower - upper) + upper
        y = self.f(X) + np.random.randn(n)*std 
        return X.reshape(-1,1),y.reshape(-1,1)

class ClassificationDataGenerator2(): 
    """ClassificationDataGenerator 

    Create 2-D toy data for classification, which has 2-class.
    """
    def __init__(self,f):
        """

        Args:
            f (object) : generate 2-D data which decision boundary is given as y = f(x) 
        """
        self.f = f 
    
    def __call__(self,n = 50,x_lower = 0,x_upper = 5,y_lower = 0,y_upper = 5,encoding="onehot"): 
        """Make data

        explanatory variable is expressed (x,y). 

        Args:
            n (int) : number of data
            x_lower,x_upper (float) : generate data almost x_lower <= x <= x_upper 
            y_lower,y_upper (float) : generate data almost y_lower <= y <= y_upper 
            encoding (str) : "onehot" or "target"
        Returns:
            X (2-D array) : explanatory variable.shape = (N_samples,2)
            y (encoding = "onehot") (2-D array) : target variable,shape = (N_samples,2)
            y (encoding = "target") (1-D array) : target variable,shape = (N_samples) 
        """ 
        X1 = np.random.rand(n)*(x_upper - x_lower) + x_lower
        X2 = np.random.rand(n)*(y_upper - y_lower) + y_lower
        X = np.concatenate([X1.reshape(-1,1),X2.reshape(-1,1)],axis = 1) 
        if encoding == "onehot":
            y = np.zeros((n,2)) 
            y[X2 > self.f(X1),1] = 1 
            y[X2 <= self.f(X1),0] = 1 
        else:
            y = np.zeros(n) 
            y[X2 > self.f(X1)] = 1 
        return X,y 

class ClassificationDataGenerator3(): 
    """ClassificationDataGenerator 

    Create 2-D toy data for classification, which has 3-class.
    """
    def __init__(self,f1,f2):
        """

        Args:
            f1 (object) : generate 2-D data which first decision boundary is given as y = f1(x) 
            f1 (object) : generate 2-D data which second decesion boundary is given as y = f2(x)
        
        Note:
            for all x, f1(x) >= f2(x). 
        """
        self.f1 = f1
        self.f2 = f2  
    
    def __call__(self,n = 50,x_lower = 0,x_upper = 5,y_lower = 0,y_upper = 5,encoding="onehot"): 
        """Make data

        explanatory variable is expressed (x,y). 

        Args:
            n (int) : number of data
            x_lower,x_upper (float) : generate data almost x_lower <= x <= x_upper 
            y_lower,y_upper (float) : generate data almost y_lower <= y <= y_upper 
            encoding (str) : "onehot" or "target"
        Returns:
            X (2-D array) : explanatory variable.shape = (N_samples,2)
            y (encoding = "onehot") (2-D array) : target variable,shape = (N_samples,3)
            y (encoding = "target") (1-D array) : target variable,shape = (N_samples) 
        """ 
        X1 = np.random.rand(n)*(x_upper - x_lower) + x_lower
        X2 = np.random.rand(n)*(y_upper - y_lower) + y_lower
        X = np.concatenate([X1.reshape(-1,1),X2.reshape(-1,1)],axis = 1) 
        if encoding == "onehot":
            y = np.zeros((n,3)) 
            condition1 = (X2 > self.f1(X1))
            condition2 = (X2 > self.f2(X1))
            y[condition1,0] = 1 
            y[np.logical_and(np.logical_not(condition1),condition2),1] = 1
            y[np.logical_not(condition2),2] = 1
        else:
            y = np.zeros(n) 
            condition1 = (X2 > self.f1(X1))
            condition2 = (X2 > self.f2(X1))
            y[condition1] = 0 
            y[np.logical_and(np.logical_not(condition1),condition2)] = 1
            y[np.logical_not(condition2)] = 2
        return X,y 


def load_iris():
    """

    Returns:
        X (2-D array): explanatary valiable 
        y (1-D array): class 0,1,2
        
    """
    dict = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    X = []
    y = [] 
    file = __file__.rstrip("datasets.py")
    os.chdir(f"{file}/../../")
    with open("data/iris.data") as f:
        data = f.read()
    
    for line in data.split("\n"):
        # sepal length | sepal width | petal length | petal width 
        if len(line) == 0:
            continue
        sl,sw,pl,pw,cl = line.split(",")
        rec = np.array(list(map(float,(sl,sw,pl,pw))))
        cl = dict[cl]

        X.append(rec)
        y.append(cl)
    return np.array(X),np.array(y)