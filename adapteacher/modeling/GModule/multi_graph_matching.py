import torch
import torch.nn as nn
from itertools import product, combinations, chain
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from adapteacher.modeling.GModule.utils.hungarian import hungarian
from adapteacher.modeling.GModule.utils.sinkhorn import Sinkhorn
from adapteacher.modeling.GModule.utils.spectral_clustering import spectral_clustering
from adapteacher.modeling.GModule.utils.pad_tensor import pad_tensor
from adapteacher.modeling.GModule.utils.affinity import Affinity
from adapteacher.modeling.GModule.utils.attentions import MultiHeadAttention
from adapteacher.modeling.GModule.utils.losses import *
from adapteacher.modeling.GModule.utils.graph_network import Feat2Graph, GraphConvolution

class Timer:
    def __init__(self):
        self.start_time = 0
    def tic(self):
        self.start_time = time.time()
    def toc(self, str=""):
        print_helper('{:.5f}sec {}'.format(time.time()-self.start_time, str))

DEBUG=False

def print_helper(*args):
    if DEBUG:
        print(*args)


class Displacement(nn.Module):
    r"""
    Displacement Layer computes the displacement vector for each point in the source image, with its corresponding point
    (or points) in target image.

    The output is a displacement matrix constructed from all displacement vectors.
    This metric measures the shift from source point to predicted target point, and can be applied for matching
    accuracy.

    Together with displacement matrix d, this function will also return a grad_mask, which helps to filter out dummy
    nodes in practice.

    .. math::
        \mathbf{d}_i = \sum_{j \in V_2} \left( \mathbf{S}_{i, j} P_{2j} \right)- P_{1i}

    Proposed by `"Zanfir et al. Deep Learning of Graph Matching. CVPR 2018."
    <http://openaccess.thecvf.com/content_cvpr_2018/html/Zanfir_Deep_Learning_of_CVPR_2018_paper.html>`_
    """
    def __init__(self):
        super(Displacement, self).__init__()

    def forward(self, s: Tensor, P_src: Tensor, P_tgt: Tensor, ns_gt: Tensor=None):
        r"""
        :param s: :math:`(b\times n_1 \times n_2)` permutation or doubly stochastic matrix. :math:`b`: batch size.
         :math:`n_1`: number of nodes in source image. :math:`n_2`: number of nodes in target image
        :param P_src: :math:`(b\times n_1 \times 2)` point set on source image
        :param P_tgt: :math:`(b\times n_2 \times 2)` point set on target image
        :param ns_gt: :math:`(b)` number of exact pairs. We support batched instances with different number of nodes,
         therefore ``ns_gt`` is required to specify the exact number of nodes of each instance in the batch.
        :return: displacement matrix d,
            mask for dummy nodes grad_mask. If ``ns_gt=None``, it will not be calculated and None is returned.
        """
        if ns_gt is None:
            max_n = s.shape[1]
            P_src = P_src[:, 0:max_n, :]
            grad_mask = None
        else:
            grad_mask = torch.zeros_like(P_src)
            for b, n in enumerate(ns_gt):
                grad_mask[b, 0:n] = 1

        d = torch.matmul(s, P_tgt) - P_src
        return d, grad_mask 
     

class G_Universe(nn.Module):
    def __init__(self, dim=256, univ_size=256):
        super(G_Universe, self).__init__()
        self.f2g = Feat2Graph(dim)
        # self.conv1 = GraphConvolution(dim, dim)
        self.g_gene = MultiHeadAttention(dim, 1, dropout=0.1, version='v2')
        self.adapt = nn.Linear(dim, dim)
        self.affinity_layer = Affinity(dim)
        self.loss = nn.CrossEntropyLoss()
        # self.logvar = nn.Linear(hidden_dim, univ_size)  # 学习 log(方差) σ^2

        self.univ_size = univ_size
        
    def forward(self, nodes, U):
        """for intra-node operation"""
        # U_list = []
        N_list = []
        E_list = []
        for id, node in enumerate(nodes):
            # node, edge = self.f2g(node)
            # node = F.relu(self.conv1(node, edge))
            node, edge = self.g_gene([node, node, node])
            D = self.cos_similarity(node)
            D_inv = 1 / (D + 1e-8)
            edge_D = edge * D_inv

            # U = self.adapt(U)

            N_list.append(torch.mm(node, U.T)) # self.affinity_layer(node, U)
            E_list.append(edge_D)
            # U_list.append(U)
        N = torch.cat(N_list, dim=0)
        # X_gt = torch.cat(labels, dim=0)
        # loss = self.loss(M, X_gt)
        return N, E_list#loss U_list,
    
    def cos_similarity(self, nodes):
        norms = torch.norm(nodes, p=2, dim=1, keepdim=True)
        dot_products = torch.sum(nodes * nodes, dim=1, keepdim=True)
        return 1 - (dot_products / (norms ** 2))
    

class U_sup(nn.Module):
    def __init__(self, num_cls, univ_size, dim=256):
        super(U_sup, self).__init__()
        # self.device = device
        self.univ_size = univ_size
        self.U = nn.Parameter(torch.randn(univ_size, dim)+1/self.univ_size)
        # self.register_buffer('U', torch.randn(univ_size, dim)+1/univ_size)
        self.num_classes = num_cls
        self.Net_U = G_Universe(dim, univ_size)
        self.node_affinity = Affinity(256)
        self.sinkhorn = Sinkhorn(max_iter=20,
                                 tau=0.05, epsilon=1e-10, batched_operation=False)
        self.InstNorm_layer = nn.InstanceNorm2d(1)
        self.matching_loss = nn.L1Loss(reduction='sum')
        self.matching = HiPPI()
        self.mse_loss = nn.MSELoss()
        # self.momentum = 0.996

    def forward(self, nodes, labels):
        device = nodes[0].device
        ms = torch.tensor([len(label) for label in labels], dtype=torch.int, device=device)
        # mscum = torch.cumsum(ms, dim=0)
    
        U, edges = self.Net_U(nodes, self.U)
        U = self.sinkhorn(U)
        A = torch.block_diag(*edges)

        rows = []
        for li in labels:
            row_blocks = [self.build_label_wise(li, lj) for lj in labels]
            row_matrix = torch.cat(row_blocks, dim=1)
            rows.append(row_matrix)

        W = torch.cat(rows, dim=0)
        A_ = torch.matmul(torch.matmul(W.t(), A), W)
        # W, U0, ms, d,
        U_ = self.matching(A_, U, ms, self.univ_size)
        # self.U = self.momentum * self.U + (1 - self.momentum) * U
        loss = self.U_loss(U, U_.detach(), self.U)
        return loss

    def build_label_wise(self, label1, label2):
        w12 = torch.mm(self.one_hot(label1), self.one_hot(label2).t())
        return w12

    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long().cpu()-1, :].to(x.device)
    
    def U_loss(self, U, U_gt, Ue, w=0.1, lam=1e-4, epsilon=1e-5):
        return w * F.mse_loss(U, U_gt) + lam * torch.norm(Ue, p='fro')
    # def U_loss(self, U, U_gt, ms, w=0.1, lam=1e-4, epsilon=1e-5):
    #     m = len(ms)
    #     U_list = torch.split(U, list(ms), dim=0)
    #     U_list_gt = torch.split(U_gt, list(ms), dim=0)
    #     loss_consistency = 0.0
    #     for i in range(m):
    #         for j in range(m):
    #             if i == j:
    #                 continue
    #             U_i = U_list[i]
    #             # U_j = U_list[j]
    #             U_j_inv = torch.linalg.pinv(U_list[j], rcond=1e-3)
    #             I = torch.eye(ms[i], ms[j], dtype=U_list[0].dtype, device=U_list[0].device)
    #             # 计算 Frobenius 范数
    #             # diff = U_i @ U_j_inv - U_list_gt[i] @ U_list_gt[j].t()  # U_i U_j^{-1} - I    
    #             diff = U_list_gt[i] @ U_list_gt[j].t() - I
    #             # loss_consistency += F.mse_loss(diff, I)/(ms[i]*ms[j])
    #             loss_consistency += w * torch.norm(diff, p='fro') / (ms[i]*ms[j]) #** 2 p='fro'
    #             # loss_consistency += w * torch.sum(torch.abs(diff)) / (ms[i]*ms[j])
    #     return loss_consistency #+ lam*torch.norm(U, p='fro')


class GA_GM(nn.Module):
    """
    Graduated Assignment solver for
     Graph Matching, Multi-Graph Matching and Multi-Graph Matching with a Mixture of Modes.

    This operation does not support batched input, and all input tensors should not have the first batch dimension.

    Parameter: maximum iteration mgm_iter
               sinkhorn iteration sk_iter
               initial sinkhorn regularization sk_tau0
               sinkhorn regularization decaying factor sk_gamma
               minimum tau value min_tau
               convergence tolerance conv_tal
    Input: multi-graph similarity matrix W
           initial multi-matching matrix U0
           number of nodes in each graph ms
           size of universe n_univ
           (optional) projector to doubly-stochastic matrix (sinkhorn) or permutation matrix (hungarian)
    Output: multi-matching matrix U
    """
    def __init__(self, mgm_iter=(200,), cluster_iter=10, sk_iter=20, sk_tau0=(0.5,), sk_gamma=0.5, cluster_beta=(1., 0.), converge_tol=1e-5, min_tau=(1e-2,), projector0=('sinkhorn',)):
        super(GA_GM, self).__init__()
        self.mgm_iter = mgm_iter
        self.cluster_iter = cluster_iter
        self.sk_iter = sk_iter
        self.sk_tau0 = sk_tau0
        self.sk_gamma = sk_gamma
        self.cluster_beta = cluster_beta
        self.converge_tol = converge_tol
        self.min_tau = min_tau
        self.projector0 = projector0

    def forward(self, A, W, U0, ms, n_univ, quad_weight=1., cluster_quad_weight=1., num_clusters=1):
        # gradient is not required for MGM module
        W = W.detach()

        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)

        Us = []
        clusters = []

        # initialize U with no clusters
        cluster_M = torch.ones(num_graphs, num_graphs, device=A.device)
        cluster_M01 = cluster_M

        U = self.gagm(A, W, U, ms, n_univ, cluster_M, self.sk_tau0[0], self.min_tau[0], self.mgm_iter[0], self.projector0[0],
                      quad_weight=quad_weight, hung_iter=(num_clusters == 1))
        Us.append(U)

        # MGM problem
        if num_clusters == 1:
            return U, torch.zeros(num_graphs, dtype=torch.int)

        for beta, sk_tau0, min_tau, max_iter, projector0 in \
                zip(self.cluster_beta, self.sk_tau0, self.min_tau, self.mgm_iter, self.projector0):
            for i in range(self.cluster_iter):
                lastU = U

                # clustering step
                def get_alpha(scale=1., qw=1.):
                    Alpha = torch.zeros(num_graphs, num_graphs, device=A.device)
                    for idx1, idx2 in product(range(num_graphs), repeat=2):
                        if idx1 == idx2:
                            continue
                        start_x = m_indices[idx1 - 1] if idx1 != 0 else 0
                        end_x = m_indices[idx1]
                        start_y = m_indices[idx2 - 1] if idx2 != 0 else 0
                        end_y = m_indices[idx2]
                        A_i = A[start_x:end_x, start_x:end_x]
                        A_j = A[start_y:end_y, start_y:end_y]
                        W_ij = W[start_x:end_x, start_y:end_y]
                        U_i = U[start_x:end_x, :]
                        U_j = U[start_y:end_y, :]
                        X_ij = torch.mm(U_i, U_j.t())
                        Alpha_ij = torch.sum(W_ij * X_ij) \
                                   + torch.exp(-torch.norm(torch.chain_matmul(X_ij.t(), A_i, X_ij) - A_j) / scale) * qw
                        Alpha[idx1, idx2] = Alpha_ij
                    return Alpha
                Alpha = get_alpha(qw=cluster_quad_weight)

                last_cluster_M01 = cluster_M01
                cluster_v = spectral_clustering(Alpha, num_clusters, normalized=True)
                cluster_M01 = (cluster_v.unsqueeze(0) == cluster_v.unsqueeze(1)).to(dtype=Alpha.dtype)
                cluster_M = (1 - beta) * cluster_M01 + beta

                if beta == self.cluster_beta[0] and i == 0:
                    clusters.append(cluster_v)

                # matching step
                U = self.gagm(A, W, U, ms, n_univ, cluster_M, sk_tau0, min_tau, max_iter,
                              projector='hungarian' if i != 0 else projector0, quad_weight=quad_weight,
                              hung_iter=(beta == self.cluster_beta[-1]))

                print_helper('beta = {:.2f}, delta U = {:.4f}, delta M = {:.4f}'.format(beta, torch.norm(lastU - U), torch.norm(last_cluster_M01 - cluster_M01)))

                Us.append(U)
                clusters.append(cluster_v)

                if beta == 1:
                    break

                if torch.norm(lastU - U) < self.converge_tol and torch.norm(last_cluster_M01 - cluster_M01) < self.converge_tol:
                    break

        #return Us, clusters
        return  U, cluster_v

    def gagm(self, A, W, U0, ms, n_univ, cluster_M, init_tau, min_tau, max_iter, projector='sinkhorn', hung_iter=True, quad_weight=1.):
        num_graphs = ms.shape[0]
        U = U0
        m_indices = torch.cumsum(ms, dim=0)

        lastU = torch.zeros_like(U)

        sinkhorn_tau = init_tau
        #beta = 0.9
        iter_flag = True

        while iter_flag:
            for i in range(max_iter):
                lastU2 = lastU
                lastU = U

                # compact matrix form update of V
                UUt = torch.mm(U, U.t())
                cluster_weight = torch.repeat_interleave(cluster_M, ms.to(dtype=torch.long), dim=0)
                cluster_weight = torch.repeat_interleave(cluster_weight, ms.to(dtype=torch.long), dim=1)
                V = torch.chain_matmul(A, UUt * cluster_weight, A, U) * quad_weight * 2 + torch.mm(W * cluster_weight, U)
                V /= num_graphs

                U_list = []
                if projector == 'hungarian':
                    m_start = 0
                    for m_end in m_indices:
                        U_list.append(hungarian(V[m_start:m_end, :n_univ]))
                        m_start = m_end
                elif projector == 'sinkhorn':
                    if torch.all(ms == ms[0]):
                        if ms[0] <= n_univ:
                            U_list.append(
                                Sinkhorn(max_iter=self.sk_iter, tau=sinkhorn_tau, batched_operation=True) \
                                    (V.reshape(num_graphs, -1, n_univ), dummy_row=True).reshape(-1, n_univ))
                        else:
                            U_list.append(
                                Sinkhorn(max_iter=self.sk_iter, tau=sinkhorn_tau, batched_operation=True) \
                                    (V.reshape(num_graphs, -1, n_univ).transpose(1, 2), dummy_row=True).transpose(1, 2).reshape(-1, n_univ))
                    else:
                        V_list = []
                        n1 = []
                        m_start = 0
                        for m_end in m_indices:
                            V_list.append(V[m_start:m_end, :n_univ])
                            n1.append(m_end - m_start)
                            m_start = m_end
                        n1 = torch.tensor(n1, device=m_indices.device)
                        U = Sinkhorn(max_iter=self.sk_iter, tau=sinkhorn_tau, batched_operation=True) \
                            (torch.stack(pad_tensor(V_list), dim=0), n1, dummy_row=True)
                        m_start = 0
                        for idx, m_end in enumerate(m_indices):
                            U_list.append(U[idx, :m_end - m_start, :])
                            m_start = m_end
                else:
                    raise NameError('Unknown projecter name: {}'.format(projector))

                U = torch.cat(U_list, dim=0)
                if num_graphs == 2:
                    U[:ms[0], :] = torch.eye(ms[0], n_univ, device=U.device)

                if torch.norm(U - lastU) < self.converge_tol or torch.norm(U - lastU2) == 0:
                    break

            if i == max_iter - 1: # not converged
                if hung_iter:
                    pass
                else:
                    U_list = [hungarian(_) for _ in U_list]
                    U = torch.cat(U_list, dim=0)
                    print_helper(i, 'max iter')
                    break

            # projection control
            if projector == 'hungarian':
                print_helper(i, 'hungarian')
                break
            elif sinkhorn_tau > min_tau:
                print_helper(i, sinkhorn_tau)
                sinkhorn_tau *= self.sk_gamma
            else:
                print_helper(i, sinkhorn_tau)
                if hung_iter:
                    projector = 'hungarian'
                else:
                    U_list = [hungarian(_) for _ in U_list]
                    U = torch.cat(U_list, dim=0)
                    break

        return U


class HiPPI(nn.Module):
    """
    HiPPI solver for multiple graph matching: Higher-order Projected Power Iteration in ICCV 2019

    This operation does not support batched input, and all input tensors should not have the first batch dimension.

    Parameter: maximum iteration mgm_iter
               sinkhorn iteration sk_iter
               sinkhorn regularization sk_tau
    Input: multi-graph similarity matrix W
           initial multi-matching matrix U0
           number of nodes in each graph ms
           size of universe d
           (optional) projector to doubly-stochastic matrix (sinkhorn) or permutation matrix (hungarian)
    Output: multi-matching matrix U
    """
    def __init__(self, max_iter=50, sk_iter=20, sk_tau=1/200.):
        super(HiPPI, self).__init__()
        self.max_iter = max_iter
        self.sinkhorn = Sinkhorn(max_iter=sk_iter, tau=sk_tau)
        self.hungarian = hungarian

    def forward(self, W, U0, ms, d, projector='sinkhorn'):
        num_graphs = ms.shape[0]

        U = U0
        for i in range(self.max_iter):
            lastU = U
            WU = torch.mm(W, U) #/ num_graphs
            V = torch.chain_matmul(WU, U.t(), WU) #/ num_graphs ** 2

            #V_median = torch.median(torch.flatten(V, start_dim=-2), dim=-1).values
            #V_var, V_mean = torch.var_mean(torch.flatten(V, start_dim=-2), dim=-1)
            #V = V - V_mean
            #V = V / torch.sqrt(V_var)

            #V = V / V_median

            U = []
            m_start = 0
            m_indices = torch.cumsum(ms, dim=0)
            for m_end in m_indices:
                if projector == 'sinkhorn':
                    U.append(self.sinkhorn(V[m_start:m_end, :d], dummy_row=True))
                elif projector == 'hungarian':
                    U.append(self.hungarian(V[m_start:m_end, :d]))
                else:
                    raise NameError('Unknown projector {}.'.format(projector))
                m_start = m_end
            U = torch.cat(U, dim=0)

            #print_helper('iter={}, diff={}, var={}, vmean={}, vvar={}'.format(i, torch.norm(U-lastU), torch.var(torch.sum(U, dim=0)), V_mean, V_var))

            if torch.norm(U - lastU) < 1e-5:
                print_helper(i)
                break

        return U
    
class MGM3_unsup(nn.Module):
    def __init__(self, num_cls, univ_size, dim=256):
        super(MGM3_unsup, self).__init__()
        # self.device = cfg.MODEL.DEVICE
        self.univ_size = univ_size
        self.num_classes = num_cls
        self.quad_weight = 0.5
        self.cluster_quad_weight = 1
        self.perm_loss = 'perm'
        # self.training = 'unsup' # sup or unsup

        self.node_affinity = Affinity(d=dim)
        self.InstNorm_layer = nn.InstanceNorm2d(1)
        # self.matching_cfg = 'o2o' # o2o or m2m
        self.matching_loss = nn.L1Loss(reduction='sum')
        self.intra_domain_graph = MultiHeadAttention(dim, 1, dropout=0.1, version='v2')  # Intra-domain graph aggregation
        self.sinkhorn = Sinkhorn(max_iter=20,
                                 tau=0.05, epsilon=1e-10, batched_operation=False)
        self.ga_mgmc = GA_GM(
            mgm_iter=[200], cluster_iter=10,
            sk_iter=20, sk_tau0=[0.1], sk_gamma=0.5,
            cluster_beta=[1.0,0.0],
            converge_tol=1.0e-3, min_tau=[1.0e-2], projector0=['sinkhorn', 'sinkhorn']
        )

        if self.perm_loss == 'perm':
            self.criterion = PermutationLoss()
        elif self.perm_loss == 'ce':
            self.criterion = CrossEntropyLoss()
        elif self.perm_loss == 'focal':
            self.criterion = FocalLoss(alpha=.5, gamma=0.)
        elif self.perm_loss == 'hung':
            self.criterion = PermutationLossHung()
        elif self.perm_loss == 'hamming':
            self.criterion = HammingLoss()
        
    def forward(self, nodes, labels, U):
        device = U.device
        if nodes is None or len(nodes)==1:
            return None
        
        ms = torch.tensor([len(label) for label in labels], dtype=torch.int, device=device)
        mscum = torch.cumsum(ms, dim=0)
        mssum = mscum[-1]

        A = torch.zeros(mssum.item(), mssum.item(), device=device)
        for idx, node in enumerate(nodes):
            node, adj = self._forward_intra_graph(node)
            start_idx = mscum[idx] - ms[idx]
            end_idx = mscum[idx]
            A[start_idx:end_idx, start_idx:end_idx].add_(adj[:ms[idx], :ms[idx]])
        A.fill_diagonal_(0)

        Wds = torch.zeros(mssum.item(), mssum.item(), device=device)
        iteraion = zip(range(len(ms)), nodes, labels, ms)
        count = 0
        for src, tgt in product(iteraion, repeat=2):
            src_idx, src_feat, src_label, n_src = src
            tgt_idx, tgt_feat, tgt_label, n_tgt = tgt
            if src_idx < tgt_idx:
                count += 1
                continue
            W_ij = self._forward_aff(src_feat, tgt_feat, src_label, tgt_label)
            start_x = mscum[src_idx] - n_src
            end_x = mscum[src_idx]
            start_y = mscum[tgt_idx] - n_tgt
            end_y = mscum[tgt_idx]
            W_ijb = W_ij[:n_src, :n_tgt]
            if end_y - start_y >= end_x - start_x:
                W_ij_ds = self.sinkhorn(W_ijb, dummy_row=True)
            else:
                W_ij_ds = self.sinkhorn(W_ijb.t(), dummy_row=True).t()
            Wds[start_x:end_x, start_y:end_y] += W_ij_ds
            if src_idx != tgt_idx:
                Wds[start_y:end_y, start_x:end_x] += W_ij_ds.t()

        U_list = []
        # cluster_v = []
        # U0_b = torch.full((torch.sum(ms), self.univ_size), 1 / torch.tensor(self.univ_size, dtype=torch.float), device=device)
        # U0_b += torch.randn_like(U0_b) / 1000
        U0_b = [torch.mm(node, U.T) for node in nodes]
        U0_b = torch.cat(U0_b, dim=0).detach()
        U_b, cluster = self.ga_mgmc(A, Wds, U0_b, ms, self.univ_size, self.quad_weight, self.cluster_quad_weight)

        for i in range(len(ms)):
            if i == 0:
                start_idx = 0
            else:
                start_idx = mscum[i-1]
            end_idx = mscum[i]
            U_list.append(U_b[start_idx:end_idx, :])

        sinkhorn_pairwise_preds, hungarian_pairwise_preds, multi_graph_preds, indices = \
            self.collect_intra_class_matching_wrapper(U_list, Wds, mscum, cluster.cpu().numpy().tolist())
        
        loss_dict = {}
        loss_dict.update({
            'ds_mat_list': sinkhorn_pairwise_preds,
            'perm_mat_list': hungarian_pairwise_preds,
            'gt_perm_mat_list': multi_graph_preds, # pseudo label during training
            'graph_indices': indices,
        })

        loss = 0
        if self.perm_loss == 'offset':
            d_gt, grad_mask = self.displacement(loss_dict['gt_perm_mat'], *loss_dict['Ps'], loss_dict['ns'][0])
            d_pred, _ = self.displacement(loss_dict['ds_mat'], *loss_dict['Ps'], loss_dict['ns'][0])
            loss = self.criterion(d_pred, d_gt, grad_mask)
        elif self.perm_loss in ['perm', 'ce', 'hung', 'ilp']:
            for s_pred, x_gt, (idx_src, idx_tgt) in \
                    zip(loss_dict['ds_mat_list'], loss_dict['gt_perm_mat_list'], loss_dict['graph_indices']):
                l = self.criterion(s_pred, x_gt, ms[idx_src], ms[idx_tgt])
                loss += l
            loss /= len(loss_dict['ds_mat_list'])
            
        elif self.perm_loss == 'hamming':
            loss = self.criterion(loss_dict['perm_mat'], loss_dict['gt_perm_mat'])

        return loss # 0.01 *
        
    def _forward_intra_graph(self, nodes):

        nodes, adj = self.intra_domain_graph([nodes, nodes, nodes])
        return nodes, adj
    
    def _forward_aff(self, nodes_1, nodes_2, labels_side1, labels_side2):
        
        M = self.node_affinity(nodes_1, nodes_2)
        matching_target = torch.mm(self.one_hot(labels_side1), self.one_hot(labels_side2).t())
        # matching_loss = self.matching_loss(M.sigmoid(), matching_target.float()).mean()

        return M
    
    def one_hot(self, x):
        return torch.eye(self.num_classes)[x.long().cpu()-1, :].to(x.device)
    
    def matching_gt(self, labels, idx):
        gt = []
        for id in idx:
            gt.append(torch.mm(self.one_hot(labels[id[0]]), self.one_hot(labels[id[1]]).t()))
        
        return gt
    
    @staticmethod
    def collect_intra_class_matching_wrapper(U, Wds, mscum, cls_list):
        """
        :param U: Stacked matching-to-universe matrix
        :param Wds: pairwise matching result in doubly-stochastic matrix
        :param mscum: cumsum of number of nodes in graphs
        :param cls_list: list of class information
        """
        # collect results
        pairwise_pred_s = []
        pairwise_pred_x = []
        mgm_pred_x = []
        indices = []
        unique_cls_list = set(cls_list)

        intra_class_iterator = []
        for cls in unique_cls_list:
            idx_range = np.where(np.array(cls_list) == cls)[0]
            intra_class_iterator.append(combinations(idx_range, 2))
        intra_class_iterator = chain(*intra_class_iterator)

        for idx1, idx2 in intra_class_iterator:
            start_x = mscum[idx1 - 1] if idx1 != 0 else 0
            end_x = mscum[idx1]
            start_y = mscum[idx2 - 1] if idx2 != 0 else 0
            end_y = mscum[idx2]
            if end_y - start_y >= end_x - start_x:
                s = Wds[start_x:end_x, start_y:end_y]
            else:
                s = Wds[start_y:end_y, start_x:end_x].t()

            pairwise_pred_s.append(s.unsqueeze(0))
            x = hungarian(s)
            pairwise_pred_x.append(x.unsqueeze(0))

            mgm_x = torch.mm(U[idx1], U[idx2].t())
            mgm_pred_x.append(mgm_x.unsqueeze(0))
            indices.append((idx1, idx2))

        return pairwise_pred_s, pairwise_pred_x, mgm_pred_x, indices

def concat_matrix(matrixs, indice):
    matrixs = [ma.squeeze() for ma in matrixs]
    num_blocks = 4
    block_row_sizes = [0] * num_blocks
    block_col_sizes = [0] * num_blocks

    for idx, (row, col) in enumerate(indice):
        block_row_sizes[row] = max(block_row_sizes[row], matrixs[idx].shape[0])
        block_col_sizes[col] = max(block_col_sizes[col], matrixs[idx].shape[1])

    # 计算大矩阵的总行列大小
    big_matrix_rows = sum(block_row_sizes)
    big_matrix_cols = sum(block_col_sizes)
    big_matrix = torch.zeros((big_matrix_rows, big_matrix_cols))  # 初始化大矩阵

    # 记录每个块在大矩阵中的起始位置
    row_offsets = [0] + list(torch.cumsum(torch.tensor(block_row_sizes), dim=0).numpy())
    col_offsets = [0] + list(torch.cumsum(torch.tensor(block_col_sizes), dim=0).numpy())

    for idx, (row, col) in enumerate(indice):
        r_start, r_end = row_offsets[row], row_offsets[row + 1]
        c_start, c_end = col_offsets[col], col_offsets[col + 1]

        # 获取小矩阵大小
        matrix = matrixs[idx]
        mr, mc = matrix.shape

        # 填充小矩阵
        big_matrix[r_start:r_start + mr, c_start:c_start + mc] = matrix
        # 填充对称位置（转置适配不同形状）
        transposed_matrix = matrix.T  # 转置
        c_sym_start, c_sym_end = row_offsets[col], row_offsets[col + 1]
        r_sym_start, r_sym_end = col_offsets[row], col_offsets[row + 1]

        fill_rows = min(c_sym_end - c_sym_start, transposed_matrix.shape[0])
        fill_cols = min(r_sym_end - r_sym_start, transposed_matrix.shape[1])

        big_matrix[c_sym_start:c_sym_start + fill_rows, r_sym_start:r_sym_start + fill_cols] = transposed_matrix[:fill_rows, :fill_cols]
        
    return big_matrix