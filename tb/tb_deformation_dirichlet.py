import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tb_utils
import tb_manifold
import tb_tda


class tb_deformation_dirichlet(nn.Module):
    def __init__(self, mesh_size, param_list=None, state_list=None):
        """ Uses volume loss, sparsity loss, and continuity loss (motivated by dirichlet energy) to perform deformation.
        Either provide param_list or precomputed state_list.

        Args:
            mesh_size (int): number of k points sampled along each axis.
            param_list (list, optional): list of parameters of the form [t3, t4, [band index]]
            state_list (list, optional): list of precomputed states.

        """
        super().__init__()
        if state_list is not None:
            self.state_list = state_list
            assert param_list is None, "provide either the param_list or the state_list, not both "
        else:
            assert param_list is not None, "provide either the param_list or the state_list"
            param_list.append(mesh_size)
            self.state_list = tb_utils.get_state_list(*param_list)
        self.state_list = torch.tensor(np.array(self.state_list))
        self.mesh_size = mesh_size
        self.eulerangles_t = nn.Parameter(torch.zeros(3, self.mesh_size - 1))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.eulerangles_t, -0.01, 0.01)

    def deformation(self, eulerangles_t, state_t):
        so3_t = torch.zeros((self.mesh_size - 1, 3, 3), dtype=torch.double)
        deformed_t = torch.zeros_like(state_t)
        so3_t[:, 0, 0] = torch.cos(eulerangles_t[0]) * torch.cos(eulerangles_t[2]) - torch.sin(eulerangles_t[0]) * torch.sin(eulerangles_t[2]) * torch.cos(eulerangles_t[1])
        so3_t[:, 0, 1] = - torch.cos(eulerangles_t[0]) * torch.sin(eulerangles_t[2]) - torch.cos(eulerangles_t[1]) * torch.cos(eulerangles_t[2]) * torch.sin(eulerangles_t[0])
        so3_t[:, 0, 2] = torch.sin(eulerangles_t[0]) * torch.sin(eulerangles_t[1])
        so3_t[:, 1, 0] = torch.cos(eulerangles_t[2]) * torch.sin(eulerangles_t[0]) + torch.cos(eulerangles_t[0]) * torch.cos(eulerangles_t[1]) * torch.sin(eulerangles_t[2])
        so3_t[:, 1, 1] = torch.cos(eulerangles_t[0]) * torch.cos(eulerangles_t[1]) * torch.cos(eulerangles_t[2]) - torch.sin(eulerangles_t[0]) * torch.sin(eulerangles_t[2])
        so3_t[:, 1, 2] = - torch.cos(eulerangles_t[0]) * torch.sin(eulerangles_t[1])
        so3_t[:, 2, 0] = torch.sin(eulerangles_t[1]) * torch.sin(eulerangles_t[2])
        so3_t[:, 2, 1] = torch.cos(eulerangles_t[2]) * torch.sin(eulerangles_t[1])
        so3_t[:, 2, 2] = torch.cos(eulerangles_t[1])
        deformed_t[1:] = torch.matmul(so3_t, state_t[1:])
        deformed_t[0] = state_t[0]
        return deformed_t

    def loss_v_s(self, state_t):
        loss_v = 0.
        loss_s = 0.
        deformed_state = self.deformation(self.eulerangles_t, state_t)
        for i in range(self.mesh_size - 1):
            vol = torch.sqrt(torch.abs(1. - torch.abs(torch.matmul(deformed_state[i].view(1, 3), deformed_state[i + 1]))**2))
            loss_v += vol
            loss_s += vol**2
        vol = torch.sqrt(torch.abs(1. - torch.abs(torch.matmul(deformed_state[-1].view(1, 3), deformed_state[0]))**2))
        loss_v += vol
        loss_s += vol**2
        return loss_v, loss_s

    def loss_so3(self, eulerangles_t, state_t, vol_loss):
        """ note that eulerangles_t  shorter than state_t by 1"""
        deformed_state = self.deformation(eulerangles_t, state_t)
        similarity = torch.zeros(self.mesh_size)
        average_length = abs(vol_loss / self.mesh_size)
        for i in range(self.mesh_size - 1):
            similarity[i] = torch.exp(-torch.sqrt(torch.abs(1. - torch.abs(torch.matmul(deformed_state[i].view(1, 3), deformed_state[i + 1]))**2)) / average_length)
        similarity[-1] = torch.exp(-(1. - torch.abs(torch.matmul(deformed_state[-1].view(1, 3), deformed_state[0]))**2) / average_length)
        loss_so3 = torch.abs(eulerangles_t[:, 0]) * similarity[0] + torch.abs(eulerangles_t[:, -1]) * similarity[-1]
        for i in range(self.mesh_size - 2):
            loss_so3 += torch.abs(eulerangles_t[:, i] - eulerangles_t[:, i + 1]) * similarity[i + 1]
        return torch.sum(loss_so3)

    def find_deformation(self, n_loop, ratio=1.0, ratio_s=1.0, auto_ratio=True):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        for n in range(n_loop):
            if n < 10 and auto_ratio == True:
                with torch.no_grad():
                    vol_loss, sparse_loss = self.loss_v_s(self.state_list)
                    dir_loss = self.loss_so3(self.eulerangles_t, self.state_list, vol_loss.item())
                    ratio_s = vol_loss.item() / sparse_loss.item()
                    ratio = vol_loss.item() / dir_loss.item()
            if n % 10 == 0 and auto_ratio == True and n >= 10:
                with torch.no_grad():
                    vol_loss, sparse_loss = self.loss_v_s(self.state_list)
                    dir_loss = self.loss_so3(self.eulerangles_t, self.state_list, vol_loss.item())
                    ratio_s = vol_loss.item() / sparse_loss.item()
                    if n == 10:
                        ratio = vol_loss.item() / dir_loss.item()
                        decay_rate = np.power(ratio, 0.1)
                    else:
                        ratio = max(ratio / decay_rate, 1.)
                        while ratio * dir_loss.item() / vol_loss.item() > 0.5:
                            ratio = ratio / 1.1
            vol_loss, sparse_loss = self.loss_v_s(self.state_list)
            dir_loss = self.loss_so3(self.eulerangles_t, self.state_list, vol_loss.item())
            tot_loss = vol_loss + ratio * dir_loss + ratio_s * sparse_loss
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()

    def get_deformed_state(self):
        with torch.no_grad():
            learned_deformed_state = np.array(self.deformation(self.eulerangles_t, self.state_list))
            return learned_deformed_state

    def save_deformed_manifold(self, file_name_isomap, file_name_mds, n_components):
        learned_deformed_state = self.get_deformed_state()
        manifold = tb_manifold.tb_manifold(state_list=learned_deformed_state)
        manifold.save_isomap(file_name_isomap, n_components=n_components)
        manifold.save_mds(file_name_mds, n_components=n_components)

    def save_deformed_manifold_persistence(self, file_name_persistence):
        learned_deformed_state = self.get_deformed_state()
        tda = tb_tda.tb_tda(state_list=learned_deformed_state)
        tda.save_persistence(file_name_persistence)

    def get_deformed_manifold_persistence(self):
        learned_deformed_state = self.get_deformed_state()
        tda = tb_tda.tb_tda(state_list=learned_deformed_state)
        return tda.get_persistence()


if __name__ == "__main__":
    trivial = tb_deformation_dirichlet(150, param_list=[1.0, -1.0, [0]])
    trivial_manifold = tb_manifold.tb_manifold(state_list=trivial.get_deformed_state())
    np.save('trivial_iso_pre', np.array(trivial_manifold.get_isomap_dim_list()))
    np.save('trivial_mds_pre', np.array(trivial_manifold.get_mds_dim_list()))
    trivial.save_deformed_manifold_persistence("figures_tb/deformed_dir_persistence_trivial_pre")
    trivial.save_deformed_manifold("figures_tb/deformed_dir_isomap_trivial_pre", "figures_tb/deformed_dir_mds_trivial_pre", n_components=2)
    trivial.find_deformation(3000)
    trivial.save_deformed_manifold("figures_tb/deformed_dir_isomap_trivial", "figures_tb/deformed_dir_mds_trivial", n_components=2)
    trivial.save_deformed_manifold_persistence("figures_tb/deformed_dir_persistence_trivial")
    trivial_manifold = tb_manifold.tb_manifold(state_list=trivial.get_deformed_state())
    np.save('trivial_iso', np.array(trivial_manifold.get_isomap_dim_list()))
    np.save('trivial_mds', np.array(trivial_manifold.get_mds_dim_list()))

    topological = tb_deformation_dirichlet(150, param_list=[1.0, 1.0, [0]])
    topological_manifold = tb_manifold.tb_manifold(state_list=topological.get_deformed_state())
    np.save('topological_iso_pre', np.array(topological_manifold.get_isomap_dim_list()))
    np.save('topological_mds_pre', np.array(topological_manifold.get_mds_dim_list()))
    topological.save_deformed_manifold_persistence("figures_tb/deformed_dir_persistence_topological_pre")
    topological.save_deformed_manifold("figures_tb/deformed_dir_isomap_topological_pre", "figures_tb/deformed_dir_mds_topological_pre", n_components=2)
    topological.find_deformation(3000)
    topological.save_deformed_manifold("figures_tb/deformed_dir_isomap_topological", "figures_tb/deformed_dir_mds_topological", n_components=2)
    topological.save_deformed_manifold_persistence("figures_tb/deformed_dir_persistence_topological")
    topological_manifold = tb_manifold.tb_manifold(state_list=topological.get_deformed_state())
    np.save('topological_iso', np.array(topological_manifold.get_isomap_dim_list()))
    np.save('topological_mds', np.array(topological_manifold.get_mds_dim_list()))
