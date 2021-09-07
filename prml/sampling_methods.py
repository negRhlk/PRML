"""sampling methods

    InverseFunctionSampling
    BoxMuller 
    GaussianSampling
    RejectionSampling
    ImportanceSampling 
    SIR
    MetropolistHastingsSampling 
    HybridMonteCarlo

"""

from abc import ABC,abstractclassmethod
import numpy as np 

from prml.utils.util import _log

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
    
    def calc_expectation(self,f,n = 100):
        """calc_expectation 

        Args:
            f (np.ufunc): you want to caluculate expectation of E[f]
            n (int): number of data to calculate expectation 
            
        Returns:
            E_f (float): expectation of f 

        """

        data = self._sample(n=n) 
        return f(data)/n 


class InverseFunctionSampling(BaseSampling):
    """InverseFunctionSampling

    sample data using inverse fucntion method

    """
    def __init__(self,inv_f,a=0,b=1):
        """

        Args:
            inv_f (np.ufunc): inverse function of h(y) = \int_{-\infty}^y p(x) dx  
            a,b (float): appropriate domain of inv_f(), [a,b]
        
        """
        self.inv_f = inv_f
        self.w = b - a  
        self.c = a 

    def _sample(self, n=100):
        z = np.random.rand(n)*self.w + self.c
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


class ImportanceSampling():
    """ImportanceSampling 

    Unlike other sampling methods, ImportanceSampling does not sample from 
    the distribution, but only calculate the expected value.

    """
    def __init__(self,p,f,q=None,q_sampler=None):
        """

        Args:
            p (np.ufunc): probability densiy you want to sample 
            f (np.ufunc): you want to caluculate expectation of E[f]
            q (np.ufunc): proposal density 
            q_sampler (Sampling): proposal distribution sampler

        """
        self.p = p 
        self.f = f 
        if q is None:
            self.q = lambda x:1/(2*np.pi)**0.5*np.exp(-0.5*x**2)
            self.q_sampler = GaussSampling() 
        else:
            self.q = q 
            self.q_sampler = q_sampler

    def calc_expectation(self,n=100):
        """calc_expectation 

        Args:
            n (int): number of data to calculate expectation 

        Returns:
            E_f (float): expectation of f 

        """
        
        x = self.q_sampler.sampling(n=n)
        r = self.p(x)/self.q(x)
        f = self.f(x)
        r /= r.sum() 
        return np.dot(r,f) 


class SIR(BaseSampling):
    """SIR
    """
    def __init__(self,p,q=None,q_sampler=None):
        """

        Args:
            p (np.ufunc): probability densiy you want to sample 
            q (np.ufunc): proposal density 
            q_sampler (Sampling): proposal distribution sampler

        """
        self.p = p 
        if q is None:
            self.q = lambda x:1/(2*np.pi)**0.5*np.exp(-0.5*x**2)
            self.q_sampler = GaussSampling() 
        else:
            self.q = q 
            self.q_sampler = q_sampler
    
    def _sample(self,n=100):
        x = self.q_sampler.sampling(n=n)
        r = self.p(x)/self.q(x)
        r /= r.sum() 

        # these code can be  replaced by np.random.multinomial(n,r) 
        w_cumsum = np.cumsum(r) 
        rv = np.random.rand(n) - 1e-20 # prevent chose_idx from goin to n
        chose_idx = np.searchsorted(w_cumsum,rv) 
        return x[chose_idx]


class MetropolisHastingsSampling(BaseSampling):
    """MetropolistHastingsSampling 

    one kind of MCMC 
    """
    def __init__(self,p,D,q=None,q_sampler=None,symm=False,first_discard=0.2):
        """

        Args:  
            p (np.ufunc): probability density you want to sample 
            D (int): dimension of p(x)
            q (np.ufunc): proposal density 
            q_sampler (Sampler): proposal distribution sampler
            symm (bool): if q(z1|z2) = q(z2|z1) for all z1,z2, true. 
            first_dicard (float): first sample of this rate is discarded
        
        Note:
            if q is None and dimension of p is more than 2, this causes error 

        """
        self.p = p 
        self.D = D
        if q is None:
            self.q = lambda z,z_cond:1/(2*np.pi)**(D/2)*np.exp(-0.5*np.sum((z-z_cond)**2))
            self.q_sampler = GaussSampling(D = D)
            symm = True 
        else:
            self.q = q 
            self.q_sampler = q_sampler
        self.symm = symm
        self.first_discard = first_discard
    
    def _is_accept(self,z,z_next):
        """

        Return:
            is_accept (bool): accepct or not

        """
        if self.symm:
            accept_prob = self.p(z_next)/self.p(z)
        else:
            accept_prob = self.p(z_next)*self.q(z,z_next)/(self.p(z)*self.q(z_next,z))
        a = np.random.rand()
        return a <= accept_prob

    def _sample(self,n=100):
        m = int(n/(1 - self.first_discard) + 5) # neccesay data size
        d_size = 0
        data = np.zeros((m,self.D))
        z = np.random.rand(self.D)
        while d_size < m:
            z_next = self.q_sampler.sampling(n=1)[0] + z
            if self._is_accept(z,z_next):
                data[d_size] = z_next.copy()
                z = z_next
                d_size += 1
        return data[-n:]   


class HybridMonteCarlo(BaseSampling):
    """HybridMonteCarlo

    one kind of MCMC
    """
    def __init__(self,p,D,first_decard=0.2,L=1,eps=1e-3):
        """

        Args:  
            p (np.ufunc): probability density you want to sample, not necessarily be normalized 
            D (int): dimension of p(x)
            first_dicard (float): first sample of this rate is discarded
            L (int): how many times (r,z) steps in one sampling
            eps (float): step width is lower than this
        
        Note:
            You should choose eps so carefully to sample data correctly

        """
        self.E = lambda x:-_log(p(x))
        self.D = D
        self.first_discard = first_decard
        self.L = L
        self.eps = eps 
        self.sampler = GaussSampling(D=D)
    
    def _dE(self,z):
        dE = np.zeros(self.D)
        for i in range(self.D):
            z[i] += 1e-3
            dE[i] = self.E(z) 
            z[i] -= 1e-3 
        dE = (dE - self.E(z))*1e3 
        return dE 

    def _leapfrog(self,z,r,eps):
        # symplectic euler 
        z_now,r_now = z,r
        for _ in range(self.L):
            r_half = r_now - eps/2*self._dE(z_now)
            z_next = z_now + eps*r_half 
            r_next = r_half - eps/2*self._dE(z_next)
            z_now,r_now = z_next,r_next
        return z_now,r_now
    
    def _hamiltonian(self,z,r):
        return self.E(z) + 0.5*np.sum(r**2)

    def _is_accept(self,z,r,z_next,r_next):
        H = self._hamiltonian(z,r)
        H_next = self._hamiltonian(z_next,r_next)
        return np.random.rand() <= np.exp(H - H_next)
    
    def _sample(self,n=100):
        m = int(n/(1 - self.first_discard) + 5) # neccesary data size
        d_size = 0
        data = np.zeros((m,self.D))
        z = np.random.rand(self.D)
        r = self.sampler.sampling(n=1)[0]
        while d_size < m:
            eps = np.random.rand()*2*self.eps - self.eps # random sample from (-self.eps,self,eps)
            z_next,r_next = self._leapfrog(z,r,eps) 
            if self._is_accept(z,r,z_next,r_next):
                data[d_size] = z_next.copy() 
                z = z_next
                r = r_next
                d_size += 1
            r = self.sampler.sampling(n=1)[0]
        return data[-n:]
