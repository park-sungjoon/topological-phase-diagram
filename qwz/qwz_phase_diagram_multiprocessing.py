import qwz_deformation_dirichlet as qdefd
import multiprocessing as mp
import gudhi.wasserstein as wasserstein
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import datetime
import logging
import os
os.environ["OMP_NUM_THREADS"] = "1"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)


class diagram_mp(mp.Process):
    def __init__(self, queueLock, workQueue, outQueue, k_mesh=15):
        """ Uses multiprocessing to compute persistence diagram for parameter items in workQueue.
        The computed persistence diagrams are saved to outQueue. Note that each item contains the persistence diagram and the parameter used to compute it.

        Args:
            queueLock (mp.Lock): lock object used to control access to workQueue and outQueue.
            workQueue (mp.Queue): queue object containing the parameters for which we need to compute the diagram.
            outQueue (mp.Queue): queue object to which we store the computed diagram.
            k_mesh (int, optional): size of mesh size in the momentum space (for SSH model)
        """
        super().__init__()
        self.diagram_list = []
        self.param_list = []
        self.queueLock = queueLock
        self.workQueue = workQueue
        self.k_mesh = k_mesh
        self.outQueue = outQueue

    def run(self):
        self.queueLock.acquire()
        logger.info(f"Running process {self.name}")
        self.queueLock.release()
        while True:
            self.queueLock.acquire()
            if not self.workQueue.empty():
                mu, b = self.workQueue.get()
                logger.info(f'{self.name} acquired parameter {mu}, {b}')
                self.queueLock.release()
                deformation = qdefd.qwz_deformation_dirichlet(self.k_mesh, param_list=[mu, b, [0]])
                deformation.find_deformation(300, auto_ratio=True)
                trimmed_a = deformation.trim(int(self.k_mesh**2 * 0.14), 16)
                persistence_diagram = deformation.get_deformed_manifold_persistence(trimmed_a)

                persistence_diagram = np.array([i[1] for i in persistence_diagram if i[0] == 2])  # get persistence diagram corresponding to the second homology
                self.diagram_list.append(persistence_diagram)
                self.param_list.append((mu, b))
            else:
                self.outQueue.put(zip(self.param_list, self.diagram_list))
                self.queueLock.release()
                break
        logger.info(f"Finished thread {self.name}")


def get_diagram_list(mu_a, b_a, k_mesh, num_process=4):
    """ Uses the class diagram_mp to get a list of persistence diagram.
    Args:
        mu_a, b_a (np.array): array containing the parameters in the Hamiltonian to be used for drawing the phase diagram
        k_mesh (int): number of k points to use for computing the persistence diagram along a single k axis (for 2d problem the total number of k points is k_mesh**2).
        num_process: number of process in multiprocessing

    Returns:
        list: list of parameters and the persistence diagram after deformation.
    """
    queueLock = mp.Lock()
    workQueue = mp.Queue(len(mu_a) * len(b_a))
    outQueue = mp.Queue(num_process)
    for i in mu_a:
        for j in b_a:
            workQueue.put((i, j))
    mp_list = []

    for i in range(num_process):
        mp_list.append(diagram_mp(queueLock, workQueue, outQueue, k_mesh))
    i = 0
    for process in mp_list:
        process.start()
    for process in mp_list:
        process.join()
    logger.info('end multithreading')
    full_diag_list = []
    while outQueue.empty() is False:
        full_diag_list += list(outQueue.get())

    return full_diag_list


def get_wasserstein(full_diag_list):
    """ Compute wasserstein distance between persistence diagrams
    Args:
        full_diag_list: list of persistence diagram (from get_diagram_list)

    Returns:
        tuple: list of parameters and array of wasserstein distances (whose i,j component is the distance between persistence diagram computed for the ith and the jth parameter)

    """
    wasserstein_a = np.zeros((len(full_diag_list), len(full_diag_list)))
    param_list = []
    for i in range(len(full_diag_list)):
        param_list.append(full_diag_list[i][0])
        for j in range(1, len(full_diag_list)):
            x = full_diag_list[i][1]
            y = full_diag_list[j][1]
            wasserstein_a[i, j] = wasserstein_a[j, i] = wasserstein.wasserstein_distance(x, y, order=2)
    return param_list, wasserstein_a


def get_spectrum(wasserstein_a):
    """ Computes the eigenvalue and eigenvector for random walk Laplacian.
    Args:
        wasserstein_a: array containing wasserstein distances (from get_wasserstein)

    Returns:
        tuple: eigenvalues and eigenvectors for random walk Laplacian.
    """
    adjacency = np.exp(-wasserstein_a / (np.mean(wasserstein_a) / 2.))
    degree = np.diag(adjacency.sum(axis=0))
    degree_inv_sqrt = np.diag(1. / np.sqrt(adjacency.sum(axis=0)))
    laplacian = degree - adjacency
    laplacian_sym = degree_inv_sqrt @ laplacian @ degree_inv_sqrt
    eigenval, eigenvec = np.linalg.eigh(laplacian_sym)
    eigenvec = degree_inv_sqrt @ eigenvec
    return eigenval, eigenvec


def save_img(eigenval, eigenvec, k_means_label, file_name):
    """ Saves eigenvalue spectrum and distribution of persistence diagram in the space of eigenvectors of the random walk Laplacian. 
    Here, we explicitly indicate that only the first two eigenvectors will be used (using the knowledge that k means clustering returns two clusters, see k_means).
    Args:
        eigenval, eigenvec: outputs from get_spectrum
        file_name: name of the image file.

    Returns:
        None.

    """
    logger.info('saving image')
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    ax[0].tick_params(axis='x', labelsize=20)
    ax[0].tick_params(axis='y', labelsize=20)
    ax[1].tick_params(axis='x', labelsize=20)
    ax[1].tick_params(axis='y', labelsize=20)
    c_list = []
    c_choice = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    ax[0].plot(list(range(len(eigenval))), eigenval, '.k', linewidth=1)
    ax[0].plot(list(range(len(eigenval))), [0.2 for i in range(len(eigenval))], '-.k', linewidth=1)

    num_clusters = np.max(k_means_label) + 1
    assert num_clusters < len(c_choice), 'too many clusters'
    for i in k_means_label:
        for j in range(num_clusters):
            if i == j:
                c_list.append(c_choice[j])
                break
    ax[1].scatter(eigenvec[:, 0], eigenvec[:, 1], c=c_list, linewidth=1)
    ax[1].set_xlim(eigenvec[:, 0].min() - 0.1, eigenvec[:, 0].max() + 0.1)

    name_dir = os.path.dirname(file_name)
    os.makedirs(name_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(file_name + '.pdf', format='pdf')


def k_means(eigenval, eigenvec, cutoff=0.2):
    """ computes the clusters in eigenvector space using information in eigenvalues (last step in spectral clustering)
    Args:
        eigenval (np.array): eigenvalues of random walk laplacian
        eigenvec (np.array): eigenvectors of random walk laplacian
        cutoff (float): the number of eigenvalues below the cutoff is the number of clusters used in the k means clustering

    Returns:
        tuple: output of k means clustering (cluster centers, cluster label, inertia)

    """
    n_clusters = np.sum(eigenval < cutoff)
    k_means_clustered = cluster.k_means(eigenvec[:, :n_clusters], n_clusters=n_clusters)
    return k_means_clustered


def plot_phase_diagram(param_list, k_means_label, file_name, plot_boundary=True, save_data=True):
    """ plots phase diagram for qwz model
    Args:
        param_list: list or array such that [:,0] contains list of mu and [:,1] contains list of b (outputs of get_wasersten).
        k_means_label: output of k means clustering (k_means) for the elements in param_list
        file_name: name of the file to which we save the phase diagram
    Returns:
        None

    """
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    c_list = []
    c_choice = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    num_clusters = np.max(k_means_label) + 1
    assert num_clusters < len(c_choice), 'too many clusters'

    for i in k_means_label:
        for j in range(num_clusters):
            if i == j:
                c_list.append(c_choice[j])
                break

    ax.scatter(np.array(param_list)[:, 0], np.array(param_list)[:, 1], c=c_list, linewidth=1)
    if plot_boundary == True:
        ax.plot(np.zeros(50), np.linspace(0.5, 1.5, 50), '-.k')
        ax.plot(np.zeros(50) + 8, np.linspace(0.5, 1.5, 50), '-.k')
    ax.axes.yaxis.set_ticks([1])

    name_dir = os.path.dirname(file_name)
    os.makedirs(name_dir, exist_ok=True)
    plt.savefig(file_name + '.pdf', format='pdf')


if __name__ == '__main__':
    num_process = 2
    mu_a = np.linspace(-4.0, 12.0, 50) + np.pi / np.sqrt(6543.21)
    b_a = np.linspace(1.0, 1.0, 1)
    time1 = datetime.datetime.now()
    diagram_list = get_diagram_list(mu_a, b_a, 40, num_process=num_process)
    time2 = datetime.datetime.now()
    tot_time = time2 - time1
    logger.info(f'{tot_time}')
    param_list, wasserstein_a = get_wasserstein(diagram_list)
    eigenval, eigenvec = get_spectrum(wasserstein_a)
    k_means_clustered = k_means(eigenval, eigenvec)
    save_img(eigenval, eigenvec, k_means_clustered[1], 'figures_qwz/L_rw_spectrum_dir_test')
    plot_phase_diagram(param_list, k_means_clustered[1], 'figures_qwz/qwz_phase_diagram_dir_test')
