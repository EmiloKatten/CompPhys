import numpy as np
from week03 import AtomicClusterBase2
from week07 import DescriptorBase
from scipy.spatial.distance import pdist, squareform
from scipy.special import erf

class ConvolutedHist(DescriptorBase):

    centers = np.arange(0,4.1,0.25)
    ylim = [0, 8]

    def __init__(self, sigma=0.25):
        self.sigma = sigma
        super().__init__()
    
    def __call__(self,pos):
        dist_vector = pdist(pos)

        descriptor = []
        delta_x = self.centers[1]-self.centers[0]
        for xj in self.centers:
            val = 0
            for ri in dist_vector:
                val += 0.5*(erf((xj-ri+delta_x/2)/np.sqrt(2)/self.sigma)-\
                            erf((xj-ri-delta_x/2)/np.sqrt(2)/self.sigma))
            descriptor.append(val)
        
        return descriptor

convoluted_hist_method = ConvolutedHist()

class Calculator():
    def potential_energy(self, pos):
        return np.sum(self._V(pdist(pos)))
    
    def forces(self, pos, delta=1e-6):
        N, dim = pos.shape
        forces = np.zeros(pos.shape)

        for i in range(N):
            for d in range(dim):
                # Positive displacement
                pos_forward = np.array(pos, copy=True)
                pos_forward[i, d] += delta
                e_forward = self.potential_energy(pos_forward)

                # Negative displacement
                pos_backward = np.array(pos, copy=True)
                pos_backward[i, d] -= delta
                e_backward = self.potential_energy(pos_backward)

                # Central difference
                forces[i, d] = -(e_forward - e_backward) / (2 * delta)

        return forces

class Morse(Calculator):

    r0 = 1.
    a = 2
    eps = 1.5

    def _V(self,r):
        return self.eps * ((1.0 - np.exp(-self.a*(r - self.r0)))**2 - 1.0)

class ModifiedMorse(Morse):
    lamb = 0.5
    rcut = 0.9
    tb_strength = 10

    @classmethod
    def cosine_cutoff(cls,r):
        r = np.asarray(r) # now a numpy array (without copy if it was already)
        f = np.zeros(r.shape)
        mask1 = r <= cls.rcut
        f[mask1] = 1.0
        mask2 = (r > cls.rcut) & (r < cls.rcut + cls.lamb)
        f[mask2] = 0.5 + 0.5 * np.cos(np.pi * (r[mask2] - cls.rcut) / cls.lamb)
        return f

    def tight_binding_skewness(self,pos):
        dist_vector = pdist(pos)
        dist_matrix = squareform(dist_vector)
        connectivity_matrix = self.cosine_cutoff(dist_matrix)
        np.fill_diagonal(connectivity_matrix, 0)
        eigenvalues = np.linalg.eigvalsh(connectivity_matrix)
        mean_lambda = np.mean(eigenvalues)
        third_moment = np.mean((eigenvalues - mean_lambda) ** 3)
        return third_moment

    def potential_energy(self, pos):
        morse_part = super().potential_energy(pos)
        tight_binding_part = self.tight_binding_skewness(pos)
        return morse_part + self.tb_strength * tight_binding_part


class AtomicCluster(AtomicClusterBase2):

    def __init__(self, *args, descriptor_method=convoluted_hist_method, color = 'C4', **kwargs):
        self.descriptor_method = descriptor_method
        super().__init__(*args, **kwargs)
        self.velocities = np.zeros(self.pos.shape)
        self.color = color

    def rattle_random_dynamic_atom(self,A=0.01):
        index = np.random.choice(self.indices_dynamic_atoms)
        self.pos[index,:] += + A*np.random.randn(2)

    @property
    def descriptor(self):
        return self.descriptor_method(self.pos)

    def plot_descriptor(self, ax, add_title=True, force_plot=False):
        self.descriptor_method.color = self.color
        self.descriptor_method.plot(ax, self.pos, add_title, force_plot=force_plot)

    def copy(self):
        return self.__class__(self.calc, pos=self.pos, static=self.static, init_delta=self.init_delta,
                              descriptor_method=self.descriptor_method)
    def get_draw_colors(self):
        return [self.color for s in self.static]

    @property
    def kinetic_energy(self):
        return 0.5 * np.sum(self.velocities**2)

    def set_velocities(self,velocities):
        self.velocities = np.where(self.filter,0,velocities)

    def get_velocities(self):
        return self.velocities.copy()

    def draw_title(self,ax):
        ek = self.kinetic_energy
        ep = self.potential_energy
        etot = ek + ep
        ax.set_title(f'Ek={ek:7.2f}, Ep={ep:7.2f}, E={etot:7.2f}')

    
import copy

class EarlyStopping:

    def __init__(self, patience=20, min_delta=0.0):
        """
        Parameters
        ----------
        patience : int
            Number of consecutive epochs without improvement
            before training is stopped.

        min_delta : float
            Minimum decrease in validation loss that counts
            as an improvement.
        """

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = None
        self.best_state = None
        self.best_epoch = None

        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss, model, epoch):

        if (
            self.best_loss is None
            or validation_loss < self.best_loss - self.min_delta
        ):
            # A new lowest validation loss was found
            self.best_loss = validation_loss
            self.best_epoch = epoch

            # Keep an independent copy of the model parameters
            self.best_state = copy.deepcopy(
                model.state_dict()
            )

            self.counter = 0

        else:
            # No improvement in this epoch
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
