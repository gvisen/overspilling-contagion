from scipy.stats import ncx2, norm, chi2, erlang
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pickle
import scipy

class Process:
    """Base class for stochastic processes."""

    def __init__(self, pars = {}):
        self.pars = pars
        self.jump_times = np.array([])
        self.jump_sizes = np.array([])

    def sample(self, ts):
        raise NotImplementedError("Every subclass of Process must implement the `sample()` method.")

class Sample:
    """Base class for samples of stochastic processes."""

    def __init__(self, ts, vals, jump_times = np.array([]), jump_sizes = np.array([])):
        self.ts = ts
        self.vals = vals
        self.jump_times = jump_times
        self.jump_sizes = jump_sizes

class basic_affine_process(Process):
    '''Basic affine process.
    
    The process is a square-root diffusion (aka CIR process) with jumps.
    The jumps come from an independent (homogeneous) Poisson process with
    exponentially distributed jump sizes.
    
        dX(t) = alpha*(b - X(t)) dt + sigma \sqrt{X(t)} dW(t) + dJ(t)    (1)
    
    Estimation is exact on grid points.
    
    Parameters:
    -----------
    ts : numpy.ndarray, size (n,)
        Grid points in time at which process is evaluated.
    
    initial_value : float
        Value of X(0) (has to be deterministic)
    
    alpha : float
        Mean-reversion parameter in CIR diffusion (see Eq. (1) above).
        
    b : float
        Long-run mean parameter in CIR diffusion (see Eq. (1) above).
        
    sigma : float
        Volatility parameter in CIR diffusion (see Eq. (1) above).
        
    mu : float
        Mean jump size of exponentially distributed jump sizes.
    
    l : float
        Jump arrival rate of Poisson process (i.e. mean waiting time btw events is 1/l).
    
    References:
    -----------
    The model was first introducted in [1]. This implementation is based on [2, Sec. 3.4].
    
    [1] Duffie, Garleanu - Risk and valuation of collateralized debt obligations. 
        Financial analysts journal, 2001, 57. Jg., Nr. 1, S. 41-59.    
    [2] Glasserman - Monte Carlo methods in financial engineering. 
        New York: springer, 2004.
    '''
    
    def __init__(self, initial_value=0.04, alpha=0.6, b=0.02, sigma=0.14, mu=0.1, l=0.2):
        
        self.initial_value = initial_value
        self.alpha = alpha
        self.b = b
        self.sigma = sigma
        self.mu = mu
        self.l = l
        
        if self.alpha*self.b < 0.5*(self.sigma**2):
            print('Diffusion is a.s. absorbed in zero!')
        
        self.pars = {'initial_value' : initial_value,
                     'alpha' : alpha,
                     'b' : b,
                     'sigma' : sigma,
                     'mu' : mu,
                     'l' : l} 
        
    def sample(self, ts):
        
        # Compute jump times and jump sizes
        
        first_jump_time = np.random.exponential(scale=1/self.l)
        first_jump_size = np.random.exponential(scale=self.mu)

        jump_times = [first_jump_time]
        jump_sizes = [first_jump_size]

        while jump_times[-1] < ts[-1]:
            jump_time = np.random.exponential(scale=1/self.l) + jump_times[-1]
            jump_size = np.random.exponential(scale=self.mu)
            jump_times.append(jump_time)
            jump_sizes.append(jump_size)

        # ignore last jump time/size, because it is outside the grid ts
        # and store jump times/sizes in their respective attributes
        
        self.jump_times = np.array(jump_times[:-1])
        self.jump_sizes = np.array(jump_sizes[:-1])

        # merge jump times into grid ts
        ts = np.sort(np.concatenate((ts, self.jump_times)))

        vals = np.zeros(len(ts))
        vals[0] = self.initial_value

        for i in range(1, len(ts)):

            dt = ts[i] - ts[i-1]
            c = (self.sigma**2)*((1-np.exp(-self.alpha*dt))/(4*self.alpha))
            d = (4*self.b*self.alpha)/(self.sigma**2)
            nc = vals[i-1]*((np.exp(-self.alpha*dt))/c)

            if ts[i] in self.jump_times:
                i_jump = np.nonzero(self.jump_times == ts[i])[0][0]
                vals[i] = c*ncx2.rvs(df=d, nc=nc) + self.jump_sizes[i_jump]
            else:
                vals[i] = c*ncx2.rvs(df=d, nc=nc)

        return Sample(ts, vals, self.jump_times, self.jump_sizes)


class l(Process):
    
    def __init__(self, n, S_star, phi_B=0.1, phi_A=0.1, Delta_Gamma=0.5, verbose=True):
        self.n = n # number of entities
        self.N = set(list(range(self.n))) # {0, 1, ..., n-1}
        self.S_star = S_star # complement of C (set to evaluate), must be subset of {0, ..., n-1}
        self.C_star = self.N.difference(self.S_star)
        
        self._l = None
        self._change_of_measure = None
        
        self.factor_jump_intensity = 0.2
        
        self._p_0 = 0.2
        self._Psi_0 = 0.1
        self._mu = 10.*np.ones(n) # rate of jump size at T (mean jump size is 1/mu)
        
        self._gamma = None # rate of T (mean T is 1/gamma), gets fixed later
        self._Delta_Gamma = Delta_Gamma
        self._phi_A = phi_A
        self._phi_B = phi_B
        self._lambda_1 = 1.
        self._lambda_0 = 0. 
        
        
        self.prob_default = 0.8*(1/self.n)
        # probability that a factor jump time corresponds to
        # the default time T[k] of entity k (assumed same for all k's)
        # (must be less than 1/n)
      
        self.verbose = verbose
        
        # (S,D) -> index dictionary
        self.indices_dict = {}
        counter = 0
        for k in range(len(self.S_star)+1):
            for S in self._subsets(self.S_star, k):
                for r in sorted(range(len(S)+1), reverse=True):
                    for D in self._subsets(S, r):
                        index_S = self._set_to_int(S)
                        index_D = self._set_to_int(D)
                        self.indices_dict[(index_S, index_D)] = counter
                        counter += 1
        
        # S -> index dictionary
        self.index_dict = {}
        couter = 0
        for k in range(len(self.S_star)+1):
            for S in self._subsets(self.S_star, k):
                index_S = self._set_to_int(S)
                self.index_dict[index_S] = counter
                counter += 1

        
    def _set_to_int(self, this_set):
        '''Map sets into integers.'''
        
        return int(np.sum([2**x for x in this_set]))
        
    
    def _subsets(self, this_set, cardinality):
        """Returns list of all subsets of a set with given cardinality."""
        return [set(combination) for combination in combinations(this_set, cardinality)]
    
    def _integrate(self, proc, ts):
        return np.cumsum(np.hstack((np.zeros((self.n, 1)), proc[:, :-1]*np.diff(ts))), axis=1)
        
    def sample(self, ts):
        
        factor_proc = basic_affine_process(initial_value = self._Psi_0,
                                           alpha = 0.6,
                                           mu = 1/self._mu[0], 
                                           b = 0.02,
                                           sigma = 0.14,
                                           l = self.factor_jump_intensity)
        
        self.factor = factor_proc.sample(ts)
        ts = self.factor.ts
        
        # Compute default times T[k]
        # Default times T[k] are sub-sampled from jump times in factor
        # in such a way as to keep their intensity constant
        # and equal to (self.factor_jump_intensity * self.prob_default)
        
        T = np.inf*np.ones(self.n)
        
        to_default = list(range(self.n))
        for factor_jump_time in self.factor.jump_times:
            
            prob_someone_defaults = self.prob_default*len(to_default)
            
            if np.random.uniform() <= prob_someone_defaults:
                # select one entity to default and remove it from list `to_default`
                i_to_default = to_default.pop(np.random.randint(len(to_default)))
                T[i_to_default] = factor_jump_time

        self.T = T
                
        ### Compute factor-measurable quantities
        
        # Compute gamma(k) (intensity of T(k))
        
        T_indices = [len(ts)-1 if np.isinf(T[k]) else np.where(ts == self.T[k])[0][0] for k in range(self.n)]
              
        gamma = np.zeros((self.n, len(ts)))
        
        for k in range(self.n):
            gamma[k, :T_indices[k]+1] = self.factor_jump_intensity*self.prob_default
    
        phi_A = self._phi_A*np.ones((self.n, self.n, len(ts)))
        phi_B = self._phi_B*np.ones((self.n, self.n, len(ts)))
    
        def l(S, D):            
            index_S = int(np.sum([2**x for x in S]))
            index_D = int(np.sum([2**x for x in D]))
            
            return self._l[self.indices_dict[(index_S, index_D)], :]
        
        alpha = np.tile(self._lambda_1*self.factor.vals + self._lambda_0, (self.n, 1))
        alpha_integral = self._integrate(alpha, ts)
        
        # Compute martingales p(k)        
        
        p = np.zeros((self.n, len(ts)))
        
        for k in range(self.n):
            # Compute p_t(k) for times t < T(k)
            for i in range(T_indices[k]):
                p[k, i] = (1-np.exp(-self._Delta_Gamma))*np.exp(-alpha_integral[k, i])
            p[k, T_indices[k]:] = (1-np.exp(-self._Delta_Gamma))*np.exp(-alpha_integral[k, T_indices[k]])
                       
        
        # Compute intensity lambda
    
        lambda_intensity = np.zeros((self.n, len(ts)))
        for k in range(self.n):
            for i in range(len(ts)):
                lambda_intensity[k, i] = alpha[k, i] + (1-np.exp(-self._Delta_Gamma))*(ts[i] <= T[k])*gamma[k, i]
        
        # Compute change of measure
        
        exponent_integral = self._integrate(p*np.exp(alpha_integral)*gamma*(np.outer(T, np.ones(len(ts))) >= ts), ts)
        
        self._change_of_measure = np.zeros((self.n, len(ts)))
        for k in range(self.n):
            if T[k] <= ts[-1]:
                time_index = np.nonzero(ts==T[k])[0][0]
            else:
                time_index = -1
            self._change_of_measure[k, 0] = 1.
            for i in range(1, len(ts)):
                self._change_of_measure[k, i] = np.exp(exponent_integral[k, i])*(1 - (ts[i] >= T[k])*p[k, time_index]*np.exp(alpha_integral[k, time_index]))    
                
        
        ### Compute l^{S|D} 
        # iterate over S subset of S_star (increasing order), D subset of S (decreasing order)
        
        vals = np.zeros(len(ts))
        self._l = np.zeros((3**len(self.S_star), len(ts)))

        for k in range(len(self.S_star)+1):
            for S in self._subsets(self.S_star, k):
                for r in sorted(range(len(S)+1), reverse=True):
                    for D in self._subsets(S, r):
                        
                        C = self.N.difference(S)
                        
                        if self.verbose:
                            print('S|D:', S, '|', D)
                        
                        index_S = self._set_to_int(S)
                        index_D = self._set_to_int(D)
                        index_l = self.indices_dict[(index_S, index_D)]
                        
                        self._l[index_l, 0] = 1.
                        
                        for i in range(len(ts)-1):
                            dt = ts[i+1] - ts[i]
                            
                            dns = np.zeros(self.n)
                            for m in self.N:
                                dns[m] = 1*(ts[i+1] == T[m]) - gamma[m, i]*dt*(ts[i+1] <= T[m])
                            
                            A1 = l(S, D)[i]*np.sum([lambda_intensity[j, i] for j in C])
                            
                            A2 = 0.
                            for j in S.difference(D):        
                                
                                comp = 0.
                                for k in C.union(D):
                                    comp += phi_A[k, j, i]*(k in C) + phi_A[k, j, i]*(T[k] > ts[i])*(k in D)

                                A2 += (l(S, D)[i] - l(S.difference(set([j])), D)[i] - l(S, D.union(set([j])))[i]*p[j, i]*(T[j] < ts[i]))*comp


                            A3 = 0.
                            for m in self.N:
                                if gamma[m, i] != 0.:
                                    for j in S:
                                        A3 += (T[j] < ts[i])*((j in D)*l(S, D)[i] + (j in S.difference(D))*l(S, D.union(set([j])))[i]*p[j, i])*(phi_B[m, j, i]/gamma[m, i])*dns[m]

                                                        
                            self._l[index_l, i+1] = l(S, D)[i] - (A1 + A2)*dt + A3
            
        
        vals = l(self.S_star, set())[:]*np.prod(self._change_of_measure[list(self.C_star), :], axis=0)
                                  
        return Sample(ts, vals)