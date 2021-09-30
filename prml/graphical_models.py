"""Graphical Models

chapter8
IteratedConditionModes 

Todo:
    SumProduct
    MaxSum

""" 

import numpy as np

class IteratedConditionalModes():
    def __init__(self,h=1.0,beta=1.0,gamma=1.0,max_iter=100):
        """

        Args:   
            h (float): h should be -1 or 1 
            beta (float): energy parameter which indicates strength of connection between adjacent cells.
            gamma (float): energy parameter which indicates strength of connection between original image and denoised one.
            max_iter (int): max iteration when optimizing X 

        """
        self.h = h 
        self.beta = beta 
        self.gamma = gamma 
        self.max_iter = max_iter 

    def denoise_image(self,image):
        """denoise_image

        Args:
            image (2-D array): value should be 0 or 1, which means monochrome image
        
        Returns:
            denoised_image (2-D array): denoised image

        """
        image[image == 0] = -1 
        Y = image 
        X = np.copy(image)
        n1,n2 = X.shape 

        neighbor_factor = np.sum(X[1:]*X[:-1]) + np.sum(X[:,1:]*X[:,:-1])  
        E = self.h*np.sum(X) - self.beta*neighbor_factor - self.gamma*n1*n2 
        for _ in range(self.max_iter):
            changed = 0 
            for i in range(n1):
                for j in range(n2): 
                    idxs = self._get_neighbor_index(i,j,n1,n2) 
                    delta_E = -X[i][j]*2*self.h + X[i][j]*2*self.beta*np.sum(X[idxs]) + 2*self.gamma*Y[i][j]
                    if delta_E < 0: 
                        E += delta_E 
                        changed += 1 
                        X[i][j] *= -1
            
            if changed == 0:
                break 
        
        image[image == -1] = 0
        X[X == -1] = 0
        return X 
    
    def _get_neighbor_index(self,i,j,n1,n2):
        """

        Args:
            i,j (int): index of 
            n1,n2 (int): (n1.n2) is shape of image
        
        Returns:
            idxs (array): index of adjacent cells of (i,j)

        """
        idxs = [] 
        if i+1 < n1 and j+1 < n2:
            idxs.append((i+1,j+1)) 
        if i+1 < n1 and j-1 >= 0:
            idxs.append((i+1,j-1))
        if i-1 >= 0 and j+1 < n2: 
            idxs.append((i-1,j+1)) 
        if i-1 >= 0 and j-1 >= 0:
            idxs.append((i-1,j-1)) 
        return tuple(np.array(idxs).T)