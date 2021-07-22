import numpy as np
import gudhi
import matplotlib.pyplot as plt
import qwz_utils
import os


class qwz_tda():
    def __init__(self, parameter_list=None, state_list=None):
        """ creates a class to be used for topological data analysis. Note that no deformation is used here (refer to qwz_deformation).
        Either provide the parameter_list or the state_list
        Args:
            parameter_list (list, optional): list of parameters of the form [mu, b, [band index], mesh_size]
            state_list (list, optional): list of precomputed states
        """
        if state_list is None:
            assert parameter_list is not None, "provide either the parameter list or the state list"
            state_list = qwz_utils.get_state_list(*parameter_list)
        else:
            assert parameter_list is None, "provide either the parameter list or the state list, not both"
        self.distance_matrix = []
        for i in range(len(state_list)):
            distance_list = []
            for j in range(i):
                distance_list.append(qwz_utils.distance(state_list[i], state_list[j]))
            self.distance_matrix.append(distance_list)

    def get_persistence(self):
        rips = gudhi.RipsComplex(distance_matrix=self.distance_matrix, max_edge_length=1.)
        simplex_tree = rips.create_simplex_tree(max_dimension=3)
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
        plt.savefig(file_name + '.pdf', format='pdf')


if __name__ == "__main__":
    print('running qwz_tda.py')
    topological = qwz_tda([-1.0, 1.0, [0], 15])
    topological.save_persistence('figures_qwz/qwz_persistence_trivial')
    trivial = qwz_tda([1.0, 1.0, [0], 15])
    trivial.save_persistence('figures_qwz/qwz_persistence_topological')
