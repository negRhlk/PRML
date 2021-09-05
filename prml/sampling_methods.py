"""sampling methods

    InverseFunctionSampling
    BoxMuller 
    GaussianSampling

"""

from abc import ABC,abstractclassmethod
import numpy as np 

class BaseSampling(ABC):
    def __init__(self):
        pass 

    @abstractclassmethod
    def _sample(self,n=100):
        """

        Args:
            n (int): number of data 

        Returns:
            data (array): sampled data

        """
        pass 

    def sampling(self,n = 100,shape = None):
        """

        Args:
            n (int): number of data 
            shape (Tuple[int, ...]): if shape is None, no reshaping
        
        Returns:
            data (array): sampled data, shape follows argment "shape"

        """
        if shape is not None:
            sz = 1 
            for i in range(len(shape)):
                sz *= shape[i]
            if n != sz:
                raise ValueError("n should be equal to prod(shape)")
        
        data = self._sample(n = n) 
        if shape is None:
            return data
        elif data.ndim == 1:
            return data.reshape(shape)
        else:
            return data.reshape(shape+(-1,))


class InverseFunctionSampling(BaseSampling):
    """InverseFunctionSampling

    sample data using inverse fucntion method

    """
    def __init__(self,inv_f):
        """

        Args:
            inv_f (np.ufunc): inverse function of h(y) = \int_{-\infty}^y p(x) dx  
        
        """
        self.inv_f = inv_f

    def _sample(self, n=100):
        z = np.random.rand(n)
        return self.inv_f(z)


class BoxMuller(BaseSampling):
    """BoxMuller 

    sample data which follows 1d gauss(mean=0,std=1)

    """
    def _sample(self,n = 100):
        # circle 
        Z = np.random.rand(n,2) 
        Z = 2*Z - 1 
        Z = Z[Z[:,0]**2 + Z[:,1]**2 <= 1]

        r2 = Z[:,0]**2 + Z[:,1]**2 
        norm = (-2*np.log(r2)/r2)**0.5
        Y = (norm*Z.T).ravel()

        # adjust number of data
        if len(Y) < n:
            data = self._sample(n=n-len(Y))
            return np.concatenate((Y,data))
        else:
            return Y[:n]


class _GaussSampling1D(BaseSampling):
    def __init__(self,mu=None,sigma=None):
        self.gauss_sampler = BoxMuller()
        if mu is None:
            self.mu = 0 
        else:
            self.mu = mu
        if sigma is None:
            self.sigma = 1 
        else:
            self.sigma = sigma 
    
    def _sample(self, n):
        data = self.gauss_sampler._sample(n=n)
        return self.mu + data*self.sigma 


class GaussSampling(BaseSampling):
    """GaussSampling

    sample data which follows D-dim gauss(mean=mu,std=sigma)
    if D = 1,mu=0,sigma=1 it is equal to BoxMuller

    """
    def __init__(self,D=1,mu=None,sigma=None):
        """

        Args:
            D (int): dimention of gauss
            mu (1-D array): shape = (D),mean of gaussian, if None mu = 0
            sigma (2-D array): shape = (D,D),should be positive definite,std of gaussian,if None sigma = I

        """
        if D == 1:
            self.D = 1
            self.sampler = _GaussSampling1D(mu=mu,sigma=sigma)
            return 
        
        self.D = D
        self.gauss_sampler = BoxMuller()
        if mu is None:
            self.mu = np.zeros(D)
        else:
            if mu.shape[0] != D:
                raise ValueError("mu.shape[0] should be equal to D")
            else:
                self.mu = mu

        if sigma is None:
            sigma = np.eye(D)
        elif sigma.shape[0] != D:
            raise ValueError("sigma.shape[0] should be equal to D")
        
        self.L = np.linalg.cholesky(sigma)
    
    def _sample(self,n=100):
        if self.D == 1:
            return self.sampler._sample(n=n)
        data = self.gauss_sampler._sample(n = n*self.D) 
        data = data.reshape(n,self.D)
        return self.mu + np.dot(self.L,data.T).T


class RejectionSampling(BaseSampling):
    """Rejection Sampling 

    1D ramdom variable sampler

    """
    def __init__(self,p,q=None,q_sampler=None,k=None,k_lower=None,k_upper=None):
        """

        Args:
            p (np.ufunc): probability distribution you want to sample, doesn't have to be normalized 
            q (np.ufunc): proposal distribution q(x)
            q_sampler (Sampling): proposal distribution sampler
            k (float): param for proposal distribution
            k_lower,k_upper (float): when k is None,k will be decided using x in (k_lower,k_upper)

        """
        self.p = p 
        if q is None:
            self.q = lambda x:1/(2*np.pi)**0.5*np.exp(-0.5*x**2)
            self.q_sampler = GaussSampling()
        else:
            self.q = q
            self.q_sampler = q_sampler 
        
        if k is None:
            self.k = self._find_appropriate_k(k_lower,k_upper)
        else:
            self.k = k
    
    def _find_appropriate_k(self,k_lower,k_upper):
        k_lower = -100 if k_lower is None else k_lower
        k_upper = 100 if k_upper is None else k_upper 
        k = -1e20
        for x in np.linspace(k_lower,k_upper,500):
            qx = self.q(x)
            px = self.p(x)
            qx = 1e-10 if qx==0.0 else qx
            now_k = px/qx + 1e-5
            k = max(k,now_k)
        return k      

    def _sample(self,n=100):
        sample_n = int(1.5*n)
        z = self.q_sampler._sample(sample_n)
        u = np.random.rand(sample_n)*self.k*self.q(z)
        is_valid = u <= self.p(z)
        valid_size = is_valid.astype("int").sum()
        data = z[is_valid]
        if valid_size < n:
            additive_data = self._sample(n=n-valid_size)
            return np.concatenate((data,additive_data))
        else:
            return data[:n]
        