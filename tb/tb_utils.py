import numpy as np
import numpy.linalg as LA


def tb_H(k, t3, t4):
    """ Returns the Hamiltonian of three-bands model at k with hopping amplitudes t3, and t4.

    Args:
        k (float): point in momentum space.
        t3, t4 (float): hopping parameter.

    Returns:
        np.array: the 3 by 3 Hamiltonian.
    """

    hamiltonian = np.zeros((3, 3), dtype=np.double)
    hamiltonian[0, 0] = 1.0 * np.sin(k)
    hamiltonian[0, 1] = - 1.0 - 1.0 * np.cos(k)
    hamiltonian[0, 2] = t3
    hamiltonian[1, 0] = - 1.0 - 1.0 * np.cos(k)
    hamiltonian[1, 1] = - 1.0 * np.sin(k)
    hamiltonian[1, 2] = - t3
    hamiltonian[2, 0] = t3
    hamiltonian[2, 1] = - t3
    hamiltonian[2, 2] = 2.0 * t4 * np.cos(k)
    return hamiltonian


def get_state(hamiltonian, idx_list):
    """Return energy eigenstates of hamiltonian for band index in the idx_list

    Args:
        hamiltonian (np.array): the Hamiltonian matrix
        idx_list (list): list of band indices, the lowest energy band being 0

    Returns:
        np.array: energy eigenstates (column vectors) arragned according to idx_list
    """
    eigval, eigvec = LA.eigh(hamiltonian)
    return eigvec[:, idx_list]


def distance(state1, state2):
    """Return the quantum distance between state1 and state2.

    Args:
        state1 (np.array): a quantum state
        state2 (np.array): a quantum state
    Returns:
        float: quantum distance between state1 and state2, sqrt(1-abs(<state1|state2>)**2)
    """
    return np.sqrt(abs(1.0 - abs(state1.conjugate().transpose() @ state2).item()**2))


def get_state_list(t3, t4, idx_list, mesh_size):
    """ Returns list of states
    Args:
        t3, t4 (float): parameters in the Hamiltonian
        idx_list (list): list of indices for which we create the state list
        mesh_size: number of k points (uniform mesh)

    Returns:
        list: list of states  (in idx_list) for each k point
    """
    state_list = []
    k_mesh = np.linspace(0, 2. * np.pi, num=mesh_size, endpoint=False)
    for k in k_mesh:
        state_list.append(get_state(tb_H(k, t3, t4), idx_list))
    return state_list
