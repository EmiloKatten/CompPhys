import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from math import comb
from matplotlib.patches import Circle


def combination_at_index(n, k, i):
    total = comb(n, k)

    if not 0 <= i < total:
        raise IndexError(
            f"Combination index {i} outside range 0 <= i < {total}"
        )

    result = []
    start = 0

    for position in range(k):
        remaining = k - position - 1

        for value in range(start, n - remaining):
            block_size = comb(n - value - 1, remaining)

            if i < block_size:
                result.append(value)
                start = value + 1
                break

            i -= block_size

    return tuple(result)


import numpy as np

def generate_ps(L, B):
    height = max(L, B)

    landscape = np.array([
        [x, y]
        for y in reversed(range(height))
        for x in range(L + 1 + B)
    ])

    x = landscape[:, 0]
    y = landscape[:, 1]

    remove = (
        # Remove the separating column
        (x == L)

        # Remove positions below the L x L region
        | ((x < L) & (y < height - L))

        # Remove positions below the B x B region
        | ((x > L) & (y < height - B))
    )

    return landscape[~remove]


def make_microstate_factory(N, L, B=0):

    assert 0 < N <= L**2 + B**2, '0 < N =< L**2 + B**2'

    
    class MicroState:
        lamb = 0.5
        rcut = 0.9
        tb_strength = 10

        def __init__(self, i=None):


            if i is None:
                i = np.random.randint(0, self.total)

            self.i = i
            q = combination_at_index(L**2+B**2, N, i)
            self.pos = np.array([self.ps[q[n]] for n in range(self.N)],dtype='float64')
            self.Nsub = np.count_nonzero(self.pos[:, 0] < L)
            self.init_subsystem_energy()
            self.init_energy()

        @property
        def subsystem_pos(self):
            return self.pos[self.pos[:, 0] < L]
        
        @property
        def bonds(self):
            dist_vector = pdist(self.pos)
            hist = np.histogram(dist_vector, [0, 1.2, 1.5])
            return hist[0].tolist()

        @property
        def bonds_in_subsystem(self):
            dist_vector = pdist(self.subsystem_pos)
            hist = np.histogram(dist_vector, [0, 1.2, 1.5])
            return hist[0].tolist()

        @staticmethod
        def the_energy_expression(tight_bonds, long_bonds):
            return -0.5 * tight_bonds - 0.25 * long_bonds
        
        def init_energy(self):
            bonds = self.bonds
            self.energy = self.the_energy_expression(*bonds)

        def init_subsystem_energy(self):
            bonds = self.bonds_in_subsystem
            self.subsystem_energy = self.the_energy_expression(*bonds)

        def plot(self, ax, label=None, show_bonds=True, show_atoms=True):

            height = max(self.L, self.B)

            # Left L x L region
            ax.add_patch(plt.Rectangle((0, height - self.L), self.L, self.L, color="C0", alpha=0.4, zorder=-10))

            # Right B x B region
            ax.add_patch(plt.Rectangle((self.L + 1, height - self.B), self.B, self.B, color="C0", alpha=0.4, zorder=-10))
            
            for x in range(self.L + 1 + self.B + int(self.B >0)):
                ax.axvline(x, color='lightgray', linewidth=1)
            for y in range(height + 1):
                ax.axhline(y, color='lightgray', linewidth=1)
    
            if show_atoms:
                for x, y in self.pos:
                    circle = Circle((x + 0.5, y + 0.5), radius=0.4, facecolor="white", edgecolor="none") #, zorder=10)
                    ax.add_patch(circle)
            
            if show_bonds:
                for i, j in combinations(range(len(self.pos)), 2):
                    xi, yi = self.pos[i]
                    xj, yj = self.pos[j]
                    dx = abs(xi - xj)
                    dy = abs(yi - yj)
                    if dx + dy == 1:
                        color = 'black'
                        lw = 2
                    elif dx == 1 and dy == 1:
                        color = 'orange'
                        lw = 1.5
                    else:
                        continue
                    ax.plot([xi + 0.5, xj + 0.5], [yi + 0.5, yj + 0.5],
                            '-', color=color, linewidth=lw)
    
            ax.set_xlim([-0.1, self.L+int(self.B>0)+self.B + 0.1])
            ax.set_ylim([-0.1, height+0.1])
            ax.set_aspect('equal')
            ax.axis('off')
            if label == 'Energy':
                energy = self.energy
                if self.L > 0:
                    left_label_str = f'{self.subsystem_energy:.2f}'
                    ax.text(0, 1, left_label_str, transform=ax.transAxes,
                            ha='left', va='bottom', fontsize=12)
                if self.B > 0:
                    right_label_str = f'{self.energy - self.subsystem_energy:.2f}'
                    ax.text(1, 1, right_label_str, transform=ax.transAxes,
                            ha='right', va='bottom', fontsize=12)
    
        def __repr__(self):
            return f'MicroState #{self.i} of {self.total} for N={self.N} in LxL={self.L}x{self.L}'
                
    MicroState.L = L
    MicroState.B = B
    MicroState.N = N
    MicroState.total = comb(L**2 + B**2, N)
    MicroState.ps = generate_ps(L, B)
    return MicroState


