"""Probability Distribution 

chapter2 
Binary
Multi
Gaussian1D
student's t-distribution 
Histgram
Parzen
KNearestNeighbor
KNeighborClassifier

"""

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
from math import gamma
from collections import Counter


class Binary():
    """Binary Variable

    Bern(x|mu) = mu^x (1 - mu)^(1 - x)
    Beta(mu|a,b) = Gamma(a + b)/(Gamma(a)Gamma(b)) mu^(a - 1) (1 - mu)^(b - 1)
    estimate posterior distribution for mu 

    Attributes:
        a (int) : parameter for the beta distribution
        b (int) : parameter for the beta distribution 

    """
    def __init__(self,a = 3,b = 3): 
        self.a = a
        self.b = b
    
    def fit(self,X):
        """fit 

        Args:
            X (1-D array) : binary variable (0 or 1) 

        """
        l = np.sum(X)
        m = len(X) - l 
        self.a += l 
        self.b += m
        
    def plot(self):
        """plot

        plot beta disribution which mu follows

        """
        x = np.linspace(0,1,100)
        p = np.power(x,self.a - 1)*np.power(1 - x,self.b - 1)
        p /= p.sum()*0.01
        fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
        axes.plot(x,p)
        axes.set_title(f"Beta(a={self.a},b={self.b})")
        plt.show()  


class Multi():
    """Multinomial Variables 

    p(\boldsymbol{x}|\boldsymbol{\mu}) = \Pi_{k=1}^K \mu_k^{x_k}
    Dir(\boldsymbol{\mu}|\boldsymbol{\alpha}) = \frac{\Gamma(\alpha_0)}{\Gamma(\alpha_1) \cdots \Gamma(\alpha_K)}\Pi_{k=1}^K \mu_k^{\alpha_k-1}

    Attributes:
        K (int) : number of class 
        alpha (1-D array) : parameter for dirichlet distributioin

    """
    def __init__(self,K,alpha=None):
        self.K = K 
        self.alpha = alpha if alpha else np.array([1/K]*K)
    
    def fit(self,X):
        """fit

        Maximum A Posteriori for the multinomial distribution  
        
        Args:
            X (2-D array) : multinomial variables, shape = (N_samples.K) 

        """
        N = X.shape[0]
        m = X.sum()
        self.alpha += m/N 


class Gaussian1D():
    """1-D Gaussian distribution

    Attributes:
        mu (float) : parameter for the gaussian distribution
        sigma (float) : parameter for the gaussian distribution 

    """
    def __init__(self,mu=0,sigma=1):
        self.mu = mu
        self.sigma = sigma
    
    def fit(self,X):
        """fit 

        Maximum A Posteriori for the gaussian distribution 

        Args:
            X (1-D array) : shape = (N_samples)

        """
        N = len(X)
        mu_ml = X.mean()
        sigma_ml = X.var()
        self.mu = sigma_ml/(N*self.sigma + sigma_ml)*self.mu + N*self.sigma/(N*self.sigma + sigma_ml)*mu_ml
        self.sigma = 1/(1/self.sigma + N/sigma_ml)
    
    def plot(self):
        """plot

        plot gauss disribution which mu follows

        """
        x = np.linspace(self.mu - self.sigma*1.5,self.mu + self.sigma*1.5,100)
        p = np.exp(-(x - self.mu)**2/2/self.sigma)
        p /= p.sum()*0.01
        fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
        axes.plot(x,p)
        axes.set_title(f"$Gauss(\mu={self.mu:.2f},\sigma^2={self.sigma:.2f})$")
        plt.show()     


def plot_student(mu,lamda,nu):
    """plot student's t-distribution 
    """
    x = np.linspace(mu - 1/lamda*5,mu + 1/lamda*5,100)
    p = np.power(1 + lamda*(x - mu)**2/nu,-(nu + 1)/2)
    p /= p.sum()*0.01 
    fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
    axes.plot(x,p)
    axes.set_title(f"Student's t-distribution($\mu$={mu:.2f},$\lambda$={lamda:.2f},$\\nu$={nu:.2f})")
    plt.show()


class Histgram():
    """Histgram 

    histgram density estimation method 

    Attributes:
        delta (float) : smoothing parameter 

    """
    def __init__(self,delta=0.5):
        self.delta = delta 
    
    def fit(self,X):
        """

        Args:
            X (1-D array) : shape = (N_samples) 

        """
        self.X = X 
        self.ma = max(X)
        self.mi = min(X) 
    
    def plot(self):
        """plot 
        plot histgram
        """
        fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
        bins = int((self.ma - self.mi)/self.delta) 
        axes.hist(self.X,bins = bins)
        axes.set_title(f"Hist(delta = {self.delta:.2f})")
        plt.show()


class Parzen():
    """Parzen
    Parzen estimator 

    Attributes:
        kernel (obj) : "gauss" or "hist", type of kernel (or Parzen window)
        h (float) : kind of smoothing paremeter 

    """
    def __init__(self,kernel = "gauss",h = 0.5):
        self.kernel = kernel 
        self.h = h
        
    def fit(self,X):
        """fit 

        Args:
            X (1-D array) : shape = (N_samples) 

        """
        self.X = X
        self.mi = min(X)
        self.ma = max(X)
    
    def plot(self):
        """plot 

        plot distribution

        """
        xs = np.linspace(self.mi-self.h,self.ma+self.h,100)
        p = [0]*100
        if self.kernel == "gauss":
            for i,x in enumerate(xs):
                for x_n in self.X:
                    p[i] += np.exp(-(x - x_n)**2/(2*self.h**2))
                p[i] /= pow(2*np.pi*self.h**2,1/2)
                p[i] /= len(self.X) 
            fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
            axes.plot(xs,p)
            axes.set_title(f"$Parzen(Gauss Kernel)$")
            plt.show()    
            
        elif self.kernel == "hist":
            for i,x in enumerate(xs):
                for x_n in self.X:
                    if abs(x_n - x) < self.h/2:
                        p[i] += 1
                p[i] /= self.h
                p[i] /= len(self.X) 
            fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
            axes.plot(xs,p)
            axes.set_title(f"Parzen(Hist Kernel)")
            plt.show()   


class KNearestNeighbor():
    """KNearestNeighbor

    KNearest neighbor method 
    estimate distribution

    Attributes:
        k (int): number of point in the neighbor 

    """
    def __init__(self,k=5):
        self.k = k 

    def fit(self,X):
        """fit 

        Args:
            X (1-D array) : shape = (N_samples) 

        """
        self.X = X 
        self.mi = min(X)
        self.ma = max(X)

    def plot(self):
        """plot 

        plot distribution

        """
        xs = np.linspace(self.mi-0.1,self.ma+0.1,100)
        p = [0]*100
        for i,x in enumerate(xs):
            dist = abs(self.X - x)
            dist.sort() 
            h = dist[self.k-1]
            p[i] = self.k/len(self.X)/h
        fig,axes = plt.subplots(nrows = 1,ncols = 1,figsize=(7,5))
        axes.plot(xs,p)
        axes.set_title(f"KNearestNeighbor(k={self.k})")
        plt.show()  


class KNeighborClassifier():
    """KNeighborClassifier

    classify data using K nearest neighbor 

    Args:
        k (int) : number of point in the neighbor 
        color (list) : color list for plot

    """
    def __init__(self,k=10):
        self.k = k
        self.color = ["red","blue","lightgreen","yellow","orange","purple","pink"]
        
    def fit(self,X,y):
        """fit

        Args:
            X (2-D array) : shape = (N_samples,2)
            y (1-D array) : shape = (N_samples) label-encoding

        """
        self.X = X 
        self.y = y
    
    def predict(self,X):
        """predict

        predict class label for each record in X

        Args:
            X (2-D array) : shape = (N_samples,2) 

        """
        pred_y = [0]*len(X)
        for i,(a,b) in enumerate(X):
            dist_label = [((a-x)**2+(b-y)**2,label) for (x,y),label in zip(self.X,self.y)]
            dist_label.sort() 
            k_neighbor = [label for d,label in dist_label[:self.k]]
            k_neighbor_label = Counter(k_neighbor).most_common(1)[0][0]
            pred_y[i] = k_neighbor_label
        return np.array(pred_y)
        
    def plot(self):
        """plot 

        show label 
        
        """
        cmap = ListedColormap(self.color[:len(np.unique(self.y))])
        x_min,y_min = self.X.min(axis = 0)
        x_max,y_max = self.X.max(axis = 0) 
        x_min,y_min = x_min-0.1,y_min-0.1
        x_max,y_max = x_max+0.1,y_max+0.1
        x = np.linspace(x_min,x_max,100)
        y = np.linspace(y_min,y_max,100) 
        xs,ys = np.meshgrid(x,y)
        labels = self.predict(np.array([xs.ravel(),ys.ravel()]).T)
        labels = labels.reshape(xs.shape)
        
        plt.contourf(xs,ys,labels,alpha=0.3,cmap=cmap)
        plt.xlim(x_min,x_max)
        plt.ylim(y_min,y_max)
        for idx,label in enumerate(np.unique(self.y)):
            plt.scatter(x=self.X[self.y == label,0],
                        y=self.X[self.y == label,1],
                        alpha=0.8,
                        c=self.color[idx],
                        label=label)
        plt.legend()
        plt.show()