import numpy as np 

#######################
# basis function \phi #
#######################
# Gauss 
def gauss(x,mu,s):
    #:params x: 1-D array data 
    #:prams mu,s: 1-D array \mu and \sigma 
    #:return: 1-D array \phi(x)
    phi = np.exp(-(x.reshape(-1,1) - mu)/(2*s**2))
    phi = np.concatenate(([1],phi.ravel()))
    return phi

color = ["red","blue","lightgreen","yellow","orange","purple","pink"]
