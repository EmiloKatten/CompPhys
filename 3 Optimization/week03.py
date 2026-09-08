import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import fmin

def nice_plot(ax, xlim=[-2,2], ylim=[-1,3]):
    ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))
    ax.set_aspect('equal')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid('on')

class Calculator():
    def potential_energy(self, pos):
        return np.sum(self._V(pdist(pos)))
    
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

class AtomicClusterBase():

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
        
    #@property
    #def potential_energy(self):
    #    return self.calc.potential_energy(self.pos)

    @property
    def forces(self, filter=True):
        forces = self.calc.forces(self.pos)
        if filter:
            forces = np.where(self.filter,0,forces)
        return forces
    
    #def rattle_one_pos(self,A=0.01):
    #    index = np.random.choice(self.indices_dynamic_atoms)
    #    self.pos[index,:] = self.pos[index,:] + A*np.random.randn(2)
        
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
    
class AtomicClusterBase2(AtomicClusterBase):

    @property
    def potential_energy(self):
        return self.calc.potential_energy(self.pos)

def base_plot(pos0, calculators):

    calculators = np.array([calculators]).ravel()


    N = len(calculators)
    fig, axes_orig = plt.subplots(1,N,figsize=(4*N,4))

    axes = np.array([axes_orig]).ravel()
    
    atom_index = 0

    xlim = [-1, 1]
    ylim = [-0, 2]
    for ax, calc in zip(axes,calculators):
        cluster = AtomicClusterBase2(calc,pos=pos0,static=[False, True, True])
        xs = np.linspace(*xlim, 100)
        ys = np.linspace(*ylim, 100)
        xd, yd = np.meshgrid(xs, ys)
        zd = np.zeros(xd.shape)

        for i, j in np.ndindex(xd.shape):
            cluster.pos[atom_index,0],cluster.pos[atom_index,1] = xd[i,j],yd[i,j]
            zd[i,j] = cluster.potential_energy

        levels = list(np.arange(-5, -3, 0.25)) + list(np.arange(-3, 5.001, 2.5))
        cmap = 'autumn'
        ax.contour(xd, yd, zd, levels=levels, cmap=cmap)

        color_map = ax.contourf(xd, yd, zd, levels=levels, alpha=0.5, cmap=cmap)

        fig.colorbar(color_map, ax=ax)
        ax.set_aspect('equal')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(str(calc.__class__.__name__), fontsize=10)
    return fig, axes_orig