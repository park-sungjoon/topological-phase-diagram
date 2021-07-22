import numpy as np
import gudhi
import matplotlib.pyplot as plt
import tb_utils
import os


class tb_tda():
    def __init__(self, parameter_list=None, state_list=None):
        """ Carries out topological data analysis for tb model. Takes either the state_list or parameter list as input.

        Args:
            parameter_list (list): list of parameters of the format (t3, t4, [band_index], mesh_size). Note that t3 and t4 are parameters in the tb Hamiltonian
            state_list (list): list of states to be used for tda.

        """
        if state_list is None:
            assert parameter_list is not None, "provide either the parameter_list or the state_list"
            state_list = tb_utils.get_state_list(*parameter_list)
        else:
            assert parameter_list is None, "provide either the parameter_list or the state_list, not both"

        self.distance_matrix = []
        for i in range(len(state_list)):
            distance_list = []
            for j in range(i):
                distance_list.append(tb_utils.distance(state_list[i], state_list[j]))
            self.distance_matrix.append(distance_list)

    def get_persistence(self):
        rips = gudhi.RipsComplex(distance_matrix=self.distance_matrix, max_edge_length=1)
        simplex_tree = rips.create_simplex_tree(max_dimension=2)
        diag = simplex_tree.persistence()
        return diag

    def save_persistence(self, file_name, title=None):
        diag = self.get_persistence()
        ax = gudhi.plot_persistence_diagram(diag, legend=True)
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
        plt.tight_layout()

        name_dir = os.path.dirname(file_name)
        os.makedirs(name_dir, exist_ok=True)
        plt.savefig(file_name + ".pdf", format='pdf')


if __name__ == "__main__":
    topological = tb_tda([1.0, 1.0, [0], 200])
    topological.save_persistence('figures_tb/persistence_topological')
    trivial = tb_tda([1.0, -1.0, [0], 200])
    trivial.save_persistence('figures_tb/persistence_trivial')
