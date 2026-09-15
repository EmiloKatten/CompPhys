import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fmin

def nice_plot(ax):

    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.set_aspect('equal')
    ax.set_xlim([-2,2])
    ax.set_ylim([-1,2])
    ax.grid('on')

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

class AtomicCluster():

    def __init__(self, calc, N=None, pos=None, static=None, init_delta=4):
        self.calc = calc
        self.init_delta = init_delta
        assert (N is not None and pos is None) or \
               (N is None and pos is not None), 'You must specify either N or pos'
        if pos is not None:
            self.pos = np.array(pos)*1.
            self.N = len(pos)
        else:
            self.N = N
            self.pos = init_delta*np.random.rand(N,2) - init_delta/2
        if static is not None:
            assert len(static) == self.N, 'static must be N long'
            self.static = static
        else:
            self.static = [False for _ in range(self.N)]
        self.filter = np.array([self.static,self.static]).T
        self.indices_dynamic_atoms = \
                            [i for i,static in enumerate(self.static) if not static]
        self.plot_artists = {}

    def copy(self):
        return self.__class__(self.calc, pos=self.pos, static=self.static, init_delta=self.init_delta)
        
    @property
    def potential_energy(self):
        return self.calc.potential_energy(self.pos)

    @property
    def forces(self):
        forces = self.calc.forces(self.pos)
        return np.where(self.filter,0,forces)
    
    def rattle_one_pos(self,A=0.01):
        index = np.random.choice(self.indices_dynamic_atoms)
        self.pos[index,:] = self.pos[index,:] + A*np.random.randn(2)
        
    def set_positions(self,pos):
        self.pos = pos
        
    def get_positions(self):
        return self.pos.copy()

    def draw_title(self,ax):
        ax.set_title(f'E={self.potential_energy:8.3f}')

    def get_draw_colors(self):
        return ['C1' if s else 'C0' for s in self.static]

    def draw(self, ax, size=0.5, alpha=0.5, force_draw=False, edge=False, draw_label=False):  # size now in data units
        if force_draw or self.plot_artists.get(ax, None) is None:
            artists_all_atoms = []
            colors = self.get_draw_colors()
            for i,((x, y), color) in enumerate(zip(self.pos, colors)):
                artists_this_atom = []
                facecolor = to_rgba(color, alpha=alpha)
                if edge:
                    edgecolor = (0,0,0,1)
                else:
                    edgecolor = (0,0,0,0.5)
                circle = patches.Circle((x, y), radius=size, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
                ax.add_patch(circle)
                artists_this_atom.append(circle)
                if draw_label:
                    text = ax.text(x,y,str(i),ha='center',va='center',color='w')
                    artists_this_atom.append(text)
                artists_all_atoms.append(artists_this_atom)
            self.plot_artists[ax] = artists_all_atoms
        else:
            for artists_all_atoms, position in zip(self.plot_artists[ax], self.pos):
                for artist_this_atom in artists_all_atoms:
                    if isinstance(artist_this_atom, patches.Circle):
                        artist_this_atom.center = position
                    elif isinstance(artist_this_atom, plt.Text):
                        artist_this_atom.set_position(position)
        self.draw_title(ax)
    

def relax(cluster,steps=100,fnorm_min=1e-3, xtol=1e-6, ftol=1e-6):
    
    test = cluster.copy()
    def energy_of_alpha(alpha,p):
        test.set_positions(cluster.get_positions() + alpha * p)
        return test.potential_energy
    
    for i in range(steps):
        f = cluster.forces
        fnorm = np.linalg.norm(f)
        if fnorm < fnorm_min:
            return
        p = f/fnorm
        
        alpha_opt = fmin(lambda alpha: energy_of_alpha(alpha,p), 0.1, disp=False, xtol=xtol, ftol=ftol)
        cluster.set_positions(cluster.get_positions() + alpha_opt * p)
