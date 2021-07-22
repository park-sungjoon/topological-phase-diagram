import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import ssh_utils
import ssh_manifold
import ssh_tda


class ssh_deformation_dirichlet(nn.Module):
    def __init__(self, mesh_size, param_list=None, state_list=None):
        """ Uses volume loss, sparsity loss, and dirichlet loss-like function for continuity to perform deformation.
        Either provide param_list or precomputed state_list.

        Args:
            mesh_size (int): number of k points sampled along each axis.
            param_list (list, optional): list of parameters of the form [t1, t2, [band index]]
            state_list (list, optional): list of precomputed states.

        """
        super().__init__()
        if state_list is not None:
            self.state_list = state_list
            assert param_list is None, "provide either the param_list or the state_list, not both "
        else:
            assert param_list is not None, "provide either the param_list or the state_list"
            param_list.append(mesh_size)
            self.state_list = ssh_utils.get_state_list(*param_list)
        self.state_list = torch.tensor(np.array(self.state_list))
        self.mesh_size = mesh_size
        self.theta_t = nn.Parameter(torch.Tensor(self.mesh_size - 1))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.theta_t, -0.01, 0.01)

    def deformation(self, theta_t, state_t):
        so2_t = torch.zeros((self.mesh_size - 1, 2, 2), dtype=torch.double)
        deformed_t = torch.zeros_like(state_t)
        so2_t[:, 0, 0] = torch.cos(theta_t)
        so2_t[:, 1, 0] = torch.sin(theta_t)
        so2_t[:, 0, 1] = -torch.sin(theta_t)
        so2_t[:, 1, 1] = torch.cos(theta_t)
        deformed_t[1:] = torch.matmul(so2_t, state_t[1:])
        deformed_t[0] = state_t[0]
        return deformed_t

    def loss_v_s(self, state_t):
        loss_v = 0.
        loss_s = 0.
        deformed_state = self.deformation(self.theta_t, state_t)
        for i in range(self.mesh_size - 1):
            vol = torch.sqrt(torch.abs(1. - torch.abs(torch.matmul(deformed_state[i].view(1, 2), deformed_state[i + 1]))**2))
            loss_v += vol
            loss_s += vol**2
        vol = torch.sqrt(torch.abs(1. - torch.abs(torch.matmul(deformed_state[-1].view(1, 2), deformed_state[0]))**2))
        loss_v += vol
        loss_s += vol**2
        return loss_v, loss_s

    def loss_so2(self, theta_t, state_t, vol_loss):
        """ note that theta_t  shorter than state_t by 1"""
        deformed_state = self.deformation(theta_t, state_t)
        similarity = torch.zeros(self.mesh_size)
        average_length = abs(vol_loss / self.mesh_size)
        for i in range(self.mesh_size - 1):
            similarity[i] = torch.exp(-torch.sqrt(torch.abs(1. - torch.abs(torch.matmul(deformed_state[i].view(1, 2), deformed_state[i + 1]))**2)) / average_length)
        similarity[-1] = torch.exp(-(1. - torch.abs(torch.matmul(deformed_state[-1].view(1, 2), deformed_state[0]))**2) / average_length)
        loss_so2 = torch.abs(theta_t[0]) * similarity[0] + torch.abs(theta_t[-1]) * similarity[-1]
        for i in range(self.mesh_size - 2):
            loss_so2 += torch.abs(theta_t[i] - theta_t[i + 1]) * similarity[i + 1]
        return loss_so2

    def find_deformation(self, n_loop, ratio=1.0, ratio_s=1.0, auto_ratio=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for n in range(n_loop):
            if n < 10 and auto_ratio == True:
                with torch.no_grad():
                    vol_loss, sparse_loss = self.loss_v_s(self.state_list)
                    dir_loss = self.loss_so2(self.theta_t, self.state_list, vol_loss.item())
                    ratio_s = vol_loss.item() / sparse_loss.item()
                    ratio = vol_loss.item() / dir_loss.item()
            if n % 10 == 0 and auto_ratio == True and n >= 10:
                with torch.no_grad():
                    vol_loss, sparse_loss = self.loss_v_s(self.state_list)
                    dir_loss = self.loss_so2(self.theta_t, self.state_list, vol_loss.item())
                    ratio_s = vol_loss.item() / sparse_loss.item()
                    if n == 10:
                        ratio = vol_loss.item() / dir_loss.item()
                        decay_rate = np.power(ratio, 0.1)
                    else:
                        ratio = max(ratio / decay_rate, 1.)
                        while ratio * dir_loss.item() / vol_loss.item() > 0.5:
                            ratio = ratio / 1.1
                    # print(vol_loss, sparse_loss, dir_loss)
            vol_loss, sparse_loss = self.loss_v_s(self.state_list)
            dir_loss = self.loss_so2(self.theta_t, self.state_list, vol_loss.item())
            tot_loss = vol_loss + ratio * dir_loss + ratio_s * sparse_loss
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

    def get_deformed_state(self):
        with torch.no_grad():
            learned_deformed_state = np.array(self.deformation(self.theta_t, self.state_list))
            return learned_deformed_state

    def save_deformed_manifold(self, file_name_isomap, file_name_mds):
        learned_deformed_state = self.get_deformed_state()
        manifold = ssh_manifold.ssh_manifold(state_list=learned_deformed_state)
        manifold.save_isomap(file_name_isomap)
        manifold.save_mds(file_name_mds)

    def save_deformed_manifold_persistence(self, file_name_persistence):
        learned_deformed_state = self.get_deformed_state()
        tda = ssh_tda.ssh_tda(state_list=learned_deformed_state)
        tda.save_persistence(file_name_persistence)

    def get_deformed_manifold_persistence(self):
        learned_deformed_state = self.get_deformed_state()
        tda = ssh_tda.ssh_tda(state_list=learned_deformed_state)
        return tda.get_persistence()


if __name__ == "__main__":
    trivial = ssh_deformation_dirichlet(100, param_list=[1.0, 0.95, [0]])
    trivial.save_deformed_manifold_persistence("figures_ssh/ssh_deformed_dir_persistence_trivial_pre")
    trivial.save_deformed_manifold("figures_ssh/ssh_deformed_dir_isomap_trivial_pre", "figures_ssh/ssh_deformed_dir_mds_trivial_pre")
    trivial.find_deformation(510)
    trivial.save_deformed_manifold("figures_ssh/ssh_deformed_dir_isomap_trivial", "figures_ssh/ssh_deformed_dir_mds_trivial")
    trivial.save_deformed_manifold_persistence("figures_ssh/ssh_deformed_dir_persistence_trivial")
    topological = ssh_deformation_dirichlet(100, param_list=[0.95, 1.0, [0]])
    topological.save_deformed_manifold_persistence("figures_ssh/ssh_deformed_dir_persistence_topological_pre")
    topological.save_deformed_manifold("figures_ssh/ssh_deformed_dir_isomap_topological_pre", "figures_ssh/ssh_deformed_dir_mds_topological_pre")
    topological.find_deformation(510)
    topological.save_deformed_manifold("figures_ssh/ssh_deformed_dir_isomap_topological", "figures_ssh/ssh_deformed_dir_mds_topological")
    topological.save_deformed_manifold_persistence("figures_ssh/ssh_deformed_dir_persistence_topological")
