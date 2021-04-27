
import numpy as np 

EPS = 1e-20 

def _log(x): 
    """log 

    to prevent np.log_log(0), caluculate np.log(x + EPS) 

    Args:
        x (array)
    Returns:
        log (array) : same shape as x, log equals np.log(x + EPS)
    """
    if np.any(x < 0):
        print("log < 0")
        exit()
    return np.log(x + EPS) 

def sigmoid(logit): 
    """softmax function 
    Args:
        logit (array) : data
    Returns:
        prob (array) : same shape as logit 
    """
    prob = np.zeros_like(logit) 
    prob[logit >= 0] = 1/(1 + np.exp(-logit[logit >= 0])) 
    prob[logit < 0] = np.exp(logit[logit < 0])/(np.exp(logit[logit < 0]) + 1) 
    return prob 

def binary_cross_entropy(target,pred):
    """binary cross entropy

    Args:
        target (2-D array) : shape = (N_samples,1), value should be 0 or 1 
        pred (2-D array) : shape = (N_samples,1), value shoule be in (0,1)

        or 

        target (2-D array) : shape = (N_samples,2), onehotencoding 
        pred (2-D array) : shape = (N_samples,2), value shoule be int (0,1)
    
    Returns:
        loss (float) : mean of loss in records 
    """
    if target.shape[1] == 1:
        loss = -target*_log(pred) - (1.0 - target)*_log(1.0 - pred) 
    else:
        loss = target*_log(pred) 
        loss = -loss.sum(axis = 0)
    return loss.mean()  

def softmax(x):
    """softmax function 

    Args:
        x (2-D array) : shape = (N_samples,N_class) 
    
    Returns:
        prob (2-D array) : shape = (N_samples,N_class) 
    """
    
    row_max = x.max(axis = 1).reshape(-1,1)
    x -= row_max 
    ratio = np.exp(x) 
    total = ratio.sum(axis = 1).reshape(-1,1)
    return ratio/total 

def cross_entropy(target,pred):
    """cross entropy 

    Args:
        target (2-D array) : shape = (N_samples,N_class) onehotencoding 
        pred (2-D array) : shape = (N_samples,N_class) value shouled be in (0,1) 
    
    Returns:
        loss (float) : mean of loss in records 
    """
    loss = target*_log(pred) 
    return -loss.mean()

def kappa(sigma):
    """kappa 

    this is used when approximating the inverse function of a probit function 

    Args:
        sigma (array) : 
    Returns:
        kappa (array): 
    """
    return (1 + np.pi*sigma/8)**(-0.5)