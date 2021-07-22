import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import qwz_utils
import qwz_manifold
import qwz_tda
import os

class qwz_deformation_dirichlet(nn.Module):
    j = complex(0,1)
    def __init__(self, mesh_size, param_list=None, precomputed=False, state_list=None):
        """ 
        uses volume loss, sparsity loss, and naive continuity loss.
        parameter_list: mu, b, idx_list
        """
        print(torch.get_num_threads())
        super().__init__()
        if not precomputed:
            assert param_list is not None
            param_list.append(mesh_size)
            state_list = qwz_utils.get_state_list(*param_list)
        state_a_re=np.array([state.real for state in state_list])
        state_a_im=np.array([state.imag for state in state_list])
        self.state_t_re = torch.tensor(state_a_re,dtype = torch.double)
        self.state_t_im = torch.tensor(state_a_im, dtype = torch.double)
        self.mesh_size = mesh_size
        self.theta_t = nn.Parameter(torch.Tensor(self.mesh_size**2 - 1))
        self.phi_t = nn.Parameter(torch.Tensor(self.mesh_size**2 - 1))
        self.psi_t = nn.Parameter(torch.Tensor(self.mesh_size**2 - 1))
        self._init_params()

    def _init_params(self):
        nn.init.uniform_(self.theta_t, -0.0001, 0.0001)
        nn.init.uniform_(self.phi_t, -0.0001, 0.0001)
        nn.init.uniform_(self.psi_t, -0.0001, 0.0001)

    def deformation(self, theta_t, phi_t, psi_t, state_t_re, state_t_im):
        su2_t_re = torch.zeros((self.mesh_size**2 - 1, 2, 2), dtype = torch.double)
        su2_t_im = torch.zeros((self.mesh_size**2 - 1, 2, 2), dtype = torch.double)
        
        deformed_t_re = torch.zeros_like(state_t_re, dtype = torch.double)
        deformed_t_im = torch.zeros_like(state_t_im, dtype = torch.double)
        
        su2_t_re[:, 0, 0] = torch.cos(theta_t) * torch.cos(phi_t)
        su2_t_re[:, 1, 0] = torch.sin(theta_t) * torch.cos(psi_t)
        su2_t_re[:, 0, 1] = -torch.sin(theta_t) * torch.cos(psi_t)
        su2_t_re[:, 1, 1] = torch.cos(theta_t) * torch.cos(phi_t)
        
        su2_t_im[:, 0, 0] = torch.cos(theta_t) * torch.sin(phi_t)
        su2_t_im[:, 1, 0] = -torch.sin(theta_t) * torch.sin(psi_t)
        su2_t_im[:, 0, 1] = -torch.sin(theta_t) * torch.sin(psi_t)
        su2_t_im[:, 1, 1] = -torch.cos(theta_t) * torch.sin(phi_t)
        
        deformed_t_re[1:] = torch.matmul(su2_t_re, state_t_re[1:])-torch.matmul(su2_t_im, state_t_im[1:])
        deformed_t_re[0] = state_t_re[0]
        
        deformed_t_im[1:] = torch.matmul(su2_t_re, state_t_im[1:])+torch.matmul(su2_t_im, state_t_re[1:])
        deformed_t_im[0] = state_t_im[0]
        return deformed_t_re, deformed_t_im
    
    def distance(self,state1_re, state1_im, state2_re, state2_im):
        distance_HS = torch.sqrt(torch.abs(1.-(torch.matmul(state1_re.view(1,2),state2_re)+torch.matmul(state1_im.view(1,2),state2_im))**2
                  -(torch.matmul(state1_re.view(1,2),state2_im)-torch.matmul(state1_im.view(1,2),state2_re))**2))
        return distance_HS
    
    def get_distance_t(self,state_t_re, state_t_im):
        v_distance_t = torch.zeros(self.mesh_size, self.mesh_size, dtype = torch.double)
        h_distance_t = torch.zeros(self.mesh_size, self.mesh_size, dtype = torch.double)
        d_distance_t = torch.zeros(self.mesh_size, self.mesh_size, dtype = torch.double)
        for i in range(self.mesh_size-1):
            for j in range(self.mesh_size-1):
                v_distance_t[i,j] = self.distance(state_t_re[i*self.mesh_size+j],
                                                 state_t_im[i*self.mesh_size+j],
                                                 state_t_re[i*self.mesh_size+j+1],
                                                 state_t_im[i*self.mesh_size+j+1])
                h_distance_t[i,j] = self.distance(state_t_re[i*self.mesh_size+j],
                                                 state_t_im[i*self.mesh_size+j],
                                                 state_t_re[(i+1)*self.mesh_size+j],
                                                 state_t_im[(i+1)*self.mesh_size+j])
                d_distance_t[i,j] = self.distance(state_t_re[i*self.mesh_size+j],
                                                 state_t_im[i*self.mesh_size+j],
                                                 state_t_re[(i+1)*self.mesh_size+j+1],
                                                 state_t_im[(i+1)*self.mesh_size+j+1])
        for k in range(self.mesh_size):
            v_distance_t[k,-1] = self.distance(state_t_re[k*self.mesh_size+self.mesh_size-1],
                                             state_t_im[k*self.mesh_size+self.mesh_size-1],
                                             state_t_re[k*self.mesh_size],
                                             state_t_im[k*self.mesh_size])
            h_distance_t[-1,k] = self.distance(state_t_re[(self.mesh_size-1)*self.mesh_size+k],
                                             state_t_im[(self.mesh_size-1)*self.mesh_size+k],
                                             state_t_re[k],
                                             state_t_im[k])
            
        for k in range(self.mesh_size-1):
            v_distance_t[-1,k] = self.distance(state_t_re[(self.mesh_size-1)*self.mesh_size+k],
                                             state_t_im[(self.mesh_size-1)*self.mesh_size+k],
                                             state_t_re[(self.mesh_size-1)*self.mesh_size+k+1],
                                             state_t_im[(self.mesh_size-1)*self.mesh_size+k+1])
            h_distance_t[k,-1] = self.distance(state_t_re[k*self.mesh_size+self.mesh_size-1],
                                             state_t_im[k*self.mesh_size+self.mesh_size-1],
                                             state_t_re[(k+1)*self.mesh_size+self.mesh_size-1],
                                             state_t_im[(k+1)*self.mesh_size+self.mesh_size-1])
            d_distance_t[k,-1] = self.distance(state_t_re[k*self.mesh_size+self.mesh_size-1],
                                            state_t_im[k*self.mesh_size+self.mesh_size-1],
                                            state_t_re[(k+1)*self.mesh_size],
                                            state_t_im[(k+1)*self.mesh_size])
            d_distance_t[-1,k] = self.distance(state_t_re[(self.mesh_size-1)*self.mesh_size+k],
                                            state_t_im[(self.mesh_size-1)*self.mesh_size+k],
                                            state_t_re[k+1],
                                            state_t_im[k+1])
        d_distance_t[-1,-1] = self.distance(state_t_re[-1],
                                         state_t_im[-1],
                                         state_t_re[0],
                                         state_t_im[0])
        return v_distance_t, h_distance_t, d_distance_t
    
    @staticmethod
    def volume(d1, d2, d3):
        return (1./4.)*torch.sqrt(torch.abs(2.*(d1**2 * d2**2 + d1**2 * d3**2 + d2**2 * d3**2)-(d1**4+d2**4+d3**4)))
    
    def loss_v_s(self, v_distance_t, h_distance_t, d_distance_t, ratio_s):
        loss_v = 0.
        loss_s = 0.
        for i in range(self.mesh_size-1):
            for j in range(self.mesh_size-1):
                vol = self.volume(h_distance_t[i,j],v_distance_t[i+1,j],d_distance_t[i,j])
                loss_v += vol
                loss_s+=vol**2
                vol = self.volume(h_distance_t[i,j+1],v_distance_t[i,j],d_distance_t[i,j])
                loss_v += vol
                loss_s+=vol**2
                
        for k in range(self.mesh_size-1):
            vol = self.volume(h_distance_t[k,self.mesh_size-1],v_distance_t[k+1,self.mesh_size-1],d_distance_t[k,self.mesh_size-1])
            loss_v += vol
            loss_s += vol**2
            vol = self.volume(h_distance_t[k,0],v_distance_t[k,self.mesh_size-1],d_distance_t[k,self.mesh_size-1])
            loss_v += vol
            loss_s += vol**2
            
            vol = self.volume(h_distance_t[self.mesh_size-1,k],v_distance_t[0,k],d_distance_t[self.mesh_size-1,k])
            loss_v += vol
            loss_s += vol**2
            vol = self.volume(h_distance_t[self.mesh_size-1,k+1],v_distance_t[self.mesh_size-1,k],d_distance_t[self.mesh_size-1,k])
            loss_v += vol
            loss_s+= vol**2
            
        vol = self.volume(h_distance_t[self.mesh_size-1,self.mesh_size-1],v_distance_t[0,self.mesh_size-1],d_distance_t[self.mesh_size-1,self.mesh_size-1])
        loss_v += vol
        loss_s+= vol**2
        vol = self.volume(h_distance_t[self.mesh_size-1,0],v_distance_t[self.mesh_size-1,self.mesh_size-1],d_distance_t[self.mesh_size-1,self.mesh_size-1])
        loss_v += vol
        loss_s+= vol**2
        return loss_v+ratio_s * loss_s, loss_v.item(), loss_s.item()

    def loss_su2_dir(self, theta_t, phi_t, psi_t, v_distance_t, h_distance_t, tot_vol):
        # note that theta_t, phi_t, psi_t  shorter than state_t by 1
        loss_theta = 0.
        loss_phi = 0.
        loss_psi = 0.

        average_length = np.sqrt(tot_vol/(self.mesh_size**2))

        loss_theta += torch.abs(theta_t[0]) * torch.exp(-v_distance_t[0,0]/average_length) + torch.abs(theta_t[self.mesh_size-1]) * torch.exp(-h_distance_t[0,0]/average_length)
        loss_theta += torch.abs(theta_t[self.mesh_size-2]) * torch.exp(-v_distance_t[0,-1]/average_length) + torch.abs(theta_t[self.mesh_size*(self.mesh_size-1)-1]) * torch.exp(-h_distance_t[-1,0]/average_length)
        loss_phi += torch.abs(phi_t[0]) * torch.exp(-v_distance_t[0,0]/average_length) + torch.abs(phi_t[self.mesh_size-1]) * torch.exp(-h_distance_t[0,0]/average_length)
        loss_phi += torch.abs(phi_t[self.mesh_size-2]) * torch.exp(-v_distance_t[0,-1]/average_length) + torch.abs(phi_t[self.mesh_size*(self.mesh_size-1)-1]) * torch.exp(-h_distance_t[-1,0]/average_length)
        loss_psi += torch.abs(psi_t[0]) * torch.exp(-v_distance_t[0,0]/average_length) + torch.abs(psi_t[self.mesh_size-1]) * torch.exp(-h_distance_t[0,0]/average_length)
        loss_psi += torch.abs(psi_t[self.mesh_size-2]) * torch.exp(-v_distance_t[0,-1]/average_length) + torch.abs(psi_t[self.mesh_size*(self.mesh_size-1)-1]) * torch.exp(-h_distance_t[-1,0]/average_length)

        for k in range(self.mesh_size - 2):
            loss_theta += torch.abs(theta_t[k] - theta_t[k + 1]) * torch.exp(-v_distance_t[0,k+1]/average_length)
            loss_phi += torch.abs(phi_t[k] - phi_t[k + 1]) * torch.exp(-v_distance_t[0,k+1]/average_length)
            loss_psi += torch.abs(psi_t[k] - psi_t[k + 1]) * torch.exp(-v_distance_t[0,k+1]/average_length)

        for k in range(1,self.mesh_size-1):
            loss_theta += torch.abs(theta_t[k*self.mesh_size-1] - theta_t[(k+1)*self.mesh_size-1]) * torch.exp(-h_distance_t[k,0]/average_length)
            loss_phi += torch.abs(phi_t[k*self.mesh_size-1] - phi_t[(k+1)*self.mesh_size-1]) * torch.exp(-h_distance_t[k,0]/average_length)
            loss_psi += torch.abs(psi_t[k*self.mesh_size-1] - psi_t[(k+1)*self.mesh_size-1]) * torch.exp(-h_distance_t[k,0]/average_length)

        for i in range(1,self.mesh_size):
            for j in range(0, self.mesh_size-1):
                loss_theta += torch.abs(theta_t[self.mesh_size*i+j]-theta_t[self.mesh_size*i-1+j]) * torch.exp(-v_distance_t[i,j]/average_length) + torch.abs(theta_t[self.mesh_size*i+j]-theta_t[self.mesh_size*(i-1)+j]) * torch.exp(-h_distance_t[i-1,j+1]/average_length)
                loss_phi += torch.abs(phi_t[self.mesh_size*i+j]-phi_t[self.mesh_size*i-1+j]) * torch.exp(-v_distance_t[i,j]/average_length) + torch.abs(phi_t[self.mesh_size*i+j]-phi_t[self.mesh_size*(i-1)+j]) * torch.exp(-h_distance_t[i-1,j+1]/average_length)
                loss_psi += torch.abs(psi_t[self.mesh_size*i+j]-psi_t[self.mesh_size*i-1+j]) * torch.exp(-v_distance_t[i,j]/average_length) + torch.abs(psi_t[self.mesh_size*i+j]-psi_t[self.mesh_size*(i-1)+j]) * torch.exp(-h_distance_t[i-1,j+1]/average_length)
                
        for k in range(self.mesh_size-1):
            loss_theta += torch.abs(theta_t[(self.mesh_size-1)*self.mesh_size+k]-theta_t[k]) * torch.exp(-h_distance_t[-1,k+1]/average_length)
            loss_phi += torch.abs(phi_t[(self.mesh_size-1)*self.mesh_size+k]-phi_t[k]) * torch.exp(-h_distance_t[-1,k+1]/average_length)
            loss_psi += torch.abs(psi_t[(self.mesh_size-1)*self.mesh_size+k]-psi_t[k]) * torch.exp(-h_distance_t[-1,k+1]/average_length)
        
        for k in range(1,self.mesh_size):
            loss_theta += torch.abs(theta_t[self.mesh_size*k+self.mesh_size-2]-theta_t[self.mesh_size*k-1]) * torch.exp(-v_distance_t[k,-1])
            loss_phi += torch.abs(phi_t[self.mesh_size*k+self.mesh_size-2]-phi_t[self.mesh_size*k-1]) * torch.exp(-v_distance_t[k,-1])
            loss_psi += torch.abs(psi_t[self.mesh_size*k+self.mesh_size-2]-psi_t[self.mesh_size*k-1]) * torch.exp(-v_distance_t[k,-1])
        return (loss_theta+loss_phi+loss_psi)/self.mesh_size

    def find_deformation(self, n_loop, lr = 0.002, ratio=1.0, ratio_s=1.0, auto_ratio = True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for n in range(n_loop):
            if n < 50 and auto_ratio == True:
                with torch.no_grad():
                    deformed_state_re, deformed_state_im = self.deformation(self.theta_t, self.phi_t, self.psi_t, self.state_t_re, self.state_t_im)
                    v_distance_t, h_distance_t, d_distance_t = self.get_distance_t(deformed_state_re, deformed_state_im)
                    loss_v_s, tot_vol, tot_sq_vol = self.loss_v_s(v_distance_t, h_distance_t, d_distance_t, 1.)
                    ratio_s = tot_vol/tot_sq_vol
                    dir_loss = self.loss_su2_dir(self.theta_t, self.phi_t, self.psi_t, v_distance_t, h_distance_t, tot_vol)
                    ratio = tot_vol / dir_loss.item()
            if n%10 == 0 and auto_ratio == True and n >=50:
                with torch.no_grad():
                    deformed_state_re, deformed_state_im = self.deformation(self.theta_t, self.phi_t, self.psi_t, self.state_t_re, self.state_t_im)
                    v_distance_t, h_distance_t, d_distance_t = self.get_distance_t(deformed_state_re, deformed_state_im)
                    loss_v_s, tot_vol, tot_sq_vol = self.loss_v_s(v_distance_t, h_distance_t, d_distance_t, 1.)
                    ratio_s = tot_vol/tot_sq_vol
                    dir_loss = self.loss_su2_dir(self.theta_t, self.phi_t, self.psi_t, v_distance_t, h_distance_t, tot_vol)
                    if n == 50:
                        ratio = tot_vol / dir_loss.item()
                        decay_rate = np.power(ratio,0.02)
                    else:
                        ratio = max(ratio/decay_rate,1.)
                        while ratio * dir_loss.item() / tot_vol > 0.5:
                            ratio = ratio / 1.1

            deformed_state_re, deformed_state_im = self.deformation(self.theta_t, self.phi_t, self.psi_t, self.state_t_re, self.state_t_im)
            v_distance_t, h_distance_t, d_distance_t = self.get_distance_t(deformed_state_re, deformed_state_im)
            loss_v_s, tot_vol, tot_sq_vol = self.loss_v_s(v_distance_t, h_distance_t, d_distance_t, ratio_s)
            tot_loss = loss_v_s + ratio * self.loss_su2_dir(self.theta_t, self.phi_t, self.psi_t, v_distance_t, h_distance_t, tot_vol)
            optimizer.zero_grad()
            tot_loss.backward()
            optimizer.step()


    def get_deformed_state(self):
        with torch.no_grad():
            learned_deformed_state_re,learned_deformed_state_im = self.deformation(self.theta_t, self.phi_t, self.theta_t,self.state_t_re, self.state_t_im)
            return np.array(learned_deformed_state_re),np.array(learned_deformed_state_im)

    def save_deformed_manifold(self, file_name_isomap, file_name_mds,precomputed_state=False):
        if precomputed_state is not False:
            learned_state_list = precomputed_state
        else:
            learned_deformed_state_re, learned_deformed_state_im= self.get_deformed_state()
            learned_state_list = learned_deformed_state_re + self.j * learned_deformed_state_im
        manifold = qwz_manifold.qwz_manifold(state_list=learned_state_list)
        manifold.save_isomap(file_name_isomap )
        manifold.save_mds(file_name_mds)

    def save_deformed_manifold_persistence(self, file_name_persistence, precomputed_state = False):
        if precomputed_state is not False:
            learned_state_list = precomputed_state
        else:
            learned_deformed_state_re, learned_deformed_state_im= self.get_deformed_state()
            learned_state_list = learned_deformed_state_re + self.j * learned_deformed_state_im
        tda = qwz_tda.qwz_tda(state_list=learned_state_list)
        x = tda.get_persistence()
        tda.save_persistence(file_name_persistence)

    def get_deformed_manifold_persistence(self, precomputed_state = False):
        if precomputed_state is not False:
            learned_state_list = precomputed_state
        else:
            learned_deformed_state_re, learned_deformed_state_im= self.get_deformed_state()
            learned_state_list = learned_deformed_state_re + self.j * learned_deformed_state_im
        tda = qwz_tda.qwz_tda(state_list=learned_state_list)
        return tda.get_persistence()
        
    @staticmethod
    def get_distance_matrix(state_list):
        distance_matrix = np.zeros((len(state_list), len(state_list)))
        for i in range(len(state_list) - 1):
            for j in range(i + 1, len(state_list)):
                distance_matrix[i, j] = np.Inf
                distance_matrix[j, i] = qwz_utils.distance(state_list[i], state_list[j])
            distance_matrix[i,i] = np.Inf
        distance_matrix[len(state_list)-1,len(state_list)-1] = np.Inf
        return distance_matrix
    def trim(self,max_pt,size_iter):
        assert size_iter*2 +1 < max_pt, "size_iter is too large compared to max_pt"
        learned_deformed_state_re, learned_deformed_state_im= self.get_deformed_state()
        learned_state_list = learned_deformed_state_re + self.j * learned_deformed_state_im
        
        with torch.no_grad():
            deformed_state_re, deformed_state_im = self.deformation(self.theta_t, self.phi_t, self.psi_t, self.state_t_re, self.state_t_im)
            learned_state_a = learned_deformed_state_re + self.j * learned_deformed_state_im
        learned_state_list = [learned_state_a[i] for i in range(learned_state_a.shape[0])]
        distance_matrix = self.get_distance_matrix(learned_state_list)
        while len(learned_state_list) >= max_pt+size_iter:
            first_idx_arr = np.argsort(distance_matrix,axis=1)
            first_sort = np.take_along_axis(distance_matrix,first_idx_arr,axis=1)
            first_sort_mean = np.median(first_sort[:,0:10],axis=1)
            second_idx_arr = np.argsort(first_sort_mean)
            remove_idx=second_idx_arr[0:size_iter*2:2]
            learned_state_list = [learned_state_list[i] for i in range(len(learned_state_list)) if i not in remove_idx]
            distance_matrix = np.delete(distance_matrix,remove_idx,0)
            distance_matrix = np.delete(distance_matrix,remove_idx,1)
        return np.array(learned_state_list)                    
            



if __name__ == "__main__":
    trivial = qwz_deformation_dirichlet(40, param_list=[-0.1, 1.0, [0]])
    trimmed_a = trivial.trim(240,16)
    trivial.save_deformed_manifold("figures_qwz/qwz_deformed_isomap_trivial_dir_pre", "figures_qwz/qwz_deformed_mds_trivial_dir_pre",trimmed_a)
    trivial.save_deformed_manifold_persistence("figures_qwz/qwz_deformed_persistence_trivial_dir_pre", trimmed_a)
    trivial.find_deformation(550, lr=0.002, auto_ratio = True)
    trimmed_a = trivial.trim(240,16)
    trivial.save_deformed_manifold("figures_qwz/qwz_deformed_isomap_trivial_dir", "figures_qwz/qwz_deformed_mds_trivial_dir",trimmed_a)
    trivial.save_deformed_manifold_persistence("figures_qwz/qwz_deformed_persistence_trivial_dir",trimmed_a)

    topological = qwz_deformation_dirichlet(40, param_list=[0.1, 1.0, [0]])
    trimmed_a = topological.trim(240,16)
    topological.save_deformed_manifold("figures_qwz/qwz_deformed_isomap_topological_dir_pre", "figures_qwz/qwz_deformed_mds_topological_dir_pre",trimmed_a)
    topological.save_deformed_manifold_persistence("figures_qwz/qwz_deformed_persistence_topological_dir_pre", trimmed_a)
    topological.find_deformation(550, lr=0.002, auto_ratio = True)
    trimmed_a = topological.trim(240,16)
    topological.save_deformed_manifold("figures_qwz/qwz_deformed_isomap_topological_dir", "figures_qwz/qwz_deformed_mds_topological_dir",trimmed_a)
    topological.save_deformed_manifold_persistence("figures_qwz/qwz_deformed_persistence_topological_dir",trimmed_a)