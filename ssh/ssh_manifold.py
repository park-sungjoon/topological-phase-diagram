import numpy as np
import matplotlib.pyplot as plt
import ssh_utils
from sklearn.manifold import Isomap, MDS
import os


class ssh_manifold():
    def __init__(self, parameter_list=None, state_list=None):
        """ Class to be used to carry out isomap or mds. 
        We can either give the list of parameters to be used for ssh Hamiltonian or provide the states through state_list
        Args:
            parameter_list (list): list of the form [t1, t2, idx_list, mesh_size]
        """
        if state_list is None:
            assert parameter_list is not None
            state_list = ssh_utils.get_state_list(*parameter_list)
        self.distance_matrix = self.get_distance_matrix(state_list)

    def get_distance_matrix(self, state_list):
        distance_matrix = np.zeros((len(state_list), len(state_list)))
        for i in range(len(state_list) - 1):
            for j in range(i + 1, len(state_list)):
                distance_matrix[i, j] = distance_matrix[j, i] = ssh_utils.distance(state_list[i], state_list[j])
        return distance_matrix

    def get_isomap(self, n_neighbors=5, n_components=2):
        ssh_isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric='precomputed')
        isomap_out = ssh_isomap.fit_transform(self.distance_matrix)
        return isomap_out, ssh_isomap.reconstruction_error()

    def save_isomap(self, file_name, n_neighbors=5, n_components=2):
        isomap_out, reconstruction_error = self.get_isomap(n_neighbors, n_components)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.plot(isomap_out[:, 0], isomap_out[:, 1], '.', linewidth=2)
        x_min = min(isomap_out[:, 0]) * 1.2
        x_max = max(isomap_out[:, 0]) * 1.2
        y_min = min(isomap_out[:, 1]) * 1.2
        y_max = max(isomap_out[:, 1]) * 1.2
        ax.set_xlim(min(x_min, -1.), max(x_max, 1.))
        ax.set_ylim(min(y_min, -1.), max(y_max, 1.))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.tight_layout()

        name_dir = os.path.dirname(file_name)
        os.makedirs(name_dir, exist_ok=True)
        plt.savefig(file_name + ".pdf", format="pdf")
        return reconstruction_error

    def get_mds(self, n_components=2, n_init=10):
        ssh_mds = MDS(n_components=n_components, n_init=n_init, dissimilarity='precomputed')
        ssh_mds.fit(self.distance_matrix)
        mds_out = ssh_mds.embedding_
        return mds_out, ssh_mds.stress_

    def save_mds(self, file_name, n_components=2, n_init=10):
        mds_out, stress = self.get_mds(n_components, n_init)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.plot(mds_out[:, 0], mds_out[:, 1], '.', linewidth=2)
        x_min = min(mds_out[:, 0]) * 1.2
        x_max = max(mds_out[:, 0]) * 1.2
        y_min = min(mds_out[:, 1]) * 1.2
        y_max = max(mds_out[:, 1]) * 1.2
        ax.set_xlim(min(x_min, -1.), max(x_max, 1.))
        ax.set_ylim(min(y_min, -1.), max(y_max, 1.))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.tight_layout()

        name_dir = os.path.dirname(file_name)
        os.makedirs(name_dir, exist_ok=True)
        plt.savefig(file_name + ".pdf", format="pdf")
        return stress


if __name__ == "__main__":
    print('running ssh_manifold.py')
    topological = ssh_manifold([1.0, 1.5, [0], 100])
    topological.save_isomap('figures_ssh/ssh_isomap_topological')
    topological.save_mds('figures_ssh/ssh_mds_topological')
    trivial = ssh_manifold([1.5, 1.0, [0], 100])
    trivial.save_isomap('figures_ssh/ssh_isomap_trivial')
    trivial.save_mds('figures_ssh/ssh_mds_trivial')
