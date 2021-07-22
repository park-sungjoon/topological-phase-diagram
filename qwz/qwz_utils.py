import numpy as np
import numpy.linalg as LA


def qwz_hamiltonian(kx, ky, mu, b):
    """ Returns the QWZ Hamiltonian at kx,ky.

    Args:
        kx, ky (float): point in momentum space.
        mu, b (float): parameters in Hamiltonian.

    Returns:
        np.array: the QWZ Hamiltonian.
    """
    j = complex(0, 1)
    hamiltonian = np.zeros((2, 2), dtype=np.cdouble)
    hamiltonian[0, 0] = mu - 2. * b * (2. - np.cos(kx) - np.cos(ky))
    hamiltonian[1, 1] = -mu + 2. * b * (2. - np.cos(kx) - np.cos(ky))
    hamiltonian[0, 1] = np.sin(kx) - j * np.sin(ky)
    hamiltonian[1, 0] = np.sin(kx) + j * np.sin(ky)
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


def get_state_list(mu, b, idx_list, mesh_size):
    """ Returns list of states
    Args:
        mu, b (float): parameters in the Hamiltonian
        idx_list (list): list of indices for which we create the state list
        mesh_size: number of k points (uniform mesh)

    Returns:
        list: list of states  (in idx_list) for each k point.
        The state for (kx_idx, ky_idx) is located at kx_idx*mesh_size + ky_idx
    """
    state_list = []
    k_mesh = np.linspace(0, 2. * np.pi, num=mesh_size, endpoint=False)
    for kx_idx in range(mesh_size):
        for ky_idx in range(mesh_size):
            state_list.append(get_state(qwz_hamiltonian(k_mesh[kx_idx], k_mesh[ky_idx], mu, b), idx_list))
    return state_list
