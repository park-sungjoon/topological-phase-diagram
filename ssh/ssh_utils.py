import numpy as np
import numpy.linalg as LA


def ssh_hamiltonian(k, t1, t2, real_basis=True):
    """ Returns the SSH Hamiltonian at k with hopping amplitudes t1 and t2.

    Args:
        k (float): point in momentum space.
        t1 (float): hopping parameter.
        t2 (float): hopping paramter.
        real_basis (bool): if True, returns Hamiltonian in real basis. Else, use complex basis.

    Returns:
        np.array: the SSH Hamiltonian.
    """
    if real_basis:
        hamiltonian = np.zeros((2, 2), dtype=np.double)
        hamiltonian[0, 0] = t2 * np.sin(k)
        hamiltonian[1, 1] = -t2 * np.sin(k)
        hamiltonian[0, 1] = t1 + t2 * np.cos(k)
        hamiltonian[1, 0] = t1 + t2 * np.cos(k)
        return hamiltonian
    else:
        j = complex(0, 1)
        hamiltonian = np.zeros((2, 2), dtype=np.cdouble)
        hamiltonian[0, 1] = t1 + t2 * np.cos(k) - j * t2 * np.sin(k)
        hamiltonian[1, 0] = t1 + t2 * np.cos(k) + j * t2 * np.sin(k)
        return hamiltonian


def get_state(hamiltonian, idx_list):
    """Return energy eigenstates of hamiltonian for band index in the idx_list

    Args:
        Hamiltonian (np.array): the Hamiltonian matrix
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


def get_state_list(t1, t2, idx_list, mesh_size):
    """ Returns list of states
    Args:
        t1, t2 (float): parameters in the Hamiltonian
        idx_list (list): list of indices for which we create the state list
        mesh_size: number of k points (uniform mesh)

    Returns:
        list: list of states  (in idx_list) for each k point
    """
    state_list = []
    k_mesh = np.linspace(0, 2. * np.pi, num=mesh_size, endpoint=False)
    for k in k_mesh:
        state_list.append(get_state(ssh_hamiltonian(k, t1, t2), idx_list))
    return state_list
