
import numpy as np 

def softmax(logit): 
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