import numpy as np
from scipy.integrate import quad

class MonteCarlo1D():
        
    def __init__(self, potential_of_mean_force, x0=0, kT=0.15, xmin=-5, xmax=5,
                 transition_method='delta', Nsim=1000, delta=0.1):
        self.potential_of_mean_force = potential_of_mean_force
        self.x0 = x0
        self.reset_sample()
        self._kT = kT
        self.xmin = xmin
        self.xmax = xmax
        assert transition_method in ['delta','uniform'], 'Unknown transition method'
        self.transition_method = transition_method
        self.Nsim = Nsim
        self.delta = delta
    
    @property
    def kT(self):
        return self._kT
    
    @kT.setter
    def kT(self,new_kT):
        print('... changing kT')
        if self._kT != new_kT:
            self.sample = None
            self._kT = new_kT

    @property
    def sample(self):
        if self._sample is None:
            self.setup_sample()
        return self._sample

    def reset_sample(self):
        self._sample = None
    
    def estimate_from_sample(self, property):
        numerator = np.sum(property(self.sample))
        denominator = len(self.sample)
        return numerator / denominator
        
    def setup_sample(self):
        
        xs = []
        x = self.x0
        for e in range(self.Nsim):
            #if e > self.Nsim/100 and e % 1 == 0:
            xs.append(x)
            if self.transition_method == 'delta':
                x_new = x + self.delta*np.random.randn()
            else:
                x_new = self.xmin + (self.xmax-self.xmin)*np.random.rand()
            de = self.potential_of_mean_force(x_new) - self.potential_of_mean_force(x)
            if np.random.rand() < np.exp(-de/self.kT):
                x = x_new
        self._sample = np.array(xs)
        
        print('Sample size:',len(self._sample))

    def get_populations(self):
        pop_left = np.sum(np.where(self.sample<0,1,0))
        pop_right = np.sum(np.where(self.sample>=0, 1, 0))
        pops = np.array([pop_left,pop_right],dtype='float64')
        pops /= np.sum(pops)
        return pops.tolist()
        
    def plot(self,ax,xwidth=0.25, color='C0', alpha=1):
        # plot potential
        xs = np.linspace(self.xmin, self.xmax, 100)
        ax.plot(xs, self.potential_of_mean_force(xs))
        
        # plot distribution
        xs = np.arange(self.xmin,self.xmax,xwidth)
        bars, xs = np.histogram(self.sample,xs)
        
        xvals = (xs[:-1] + xs[1:]) / 2
        delta_xvals = (xvals[1] - xvals[0])
        P = bars / np.sum(bars) / delta_xvals
        width = delta_xvals * 0.8
        ax.bar(xvals, 20*P, width=width, color=color, alpha=alpha)
        
        ax.set_title(f'Simulation in A(x) at kT={self.kT}')


        Pleft, Pright = self.get_populations()
        ax.text(-4,10,'$P_{left}=$' + f'{Pleft:4.2f}', color='C1', alpha=0.8)
        ax.text(0,10,'$P_{right}=$' + f'{Pright:4.2f}', color='C1', alpha=0.8)
