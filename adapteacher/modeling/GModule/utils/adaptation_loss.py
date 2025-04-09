import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Sequence
import pdb

from torch import Tensor
import numpy as np

class ProtoLoss(nn.Module):

    """
    Official Implementaion of PCT (NIPS 2021)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int,  s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'): # device: torch.device,
        super(ProtoLoss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.prop = (torch.ones((num_classes,1))*(1/num_classes)) #.to(device)
        # self.prop = torch.tensor([[0.90],[0.10]]).to(device)
        # self.prop = torch.tensor([[0.60],[0.10],[0.05],[0.05],[0.20]]).to(device)
        # self.prop = torch.tensor([[0.50],[0.20],[0.05],[0.05],[0.20]]).to(device)
        self.eps = 1e-6
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
        # Update proportions
        self.prop = self.prop.to(mu_s.device)
        sim_mat = torch.matmul(mu_s, f_t.T)
        old_logits = self.get_pos_logits(sim_mat.detach(), self.prop)
        s_dist_old = F.softmax(old_logits, dim=0)
        prop = s_dist_old.mean(1, keepdim=True)
        # print(prop)
        self.prop = self.update_prop(prop)
        

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, self.prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        t2p_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        p2t_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*self.prop.squeeze(1)).sum()
        
        return t2p_loss, p2t_loss

class Proto_with_KLProp_Loss(nn.Module):

    """
    Official Implementaion of PCT (NIPS 2021)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(Proto_with_KLProp_Loss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.eps = 1e-6
         
    def pairwise_cosine_dist(self, x, y):
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)
        return 1 - torch.matmul(x, y.T)

    def get_pos_logits(self, sim_mat, prop):
        log_prior = torch.log(prop + self.eps)
        return sim_mat/self.nav_t + log_prior

    def update_prop(self, prop):
        return (1 - self.beta) * self.prop + self.beta * prop 

    def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor, gt_prop) -> torch.Tensor:
        # Update proportions
        sim_mat = torch.matmul(mu_s, f_t.T)

        # Calculate bi-directional transport loss
        new_logits = self.get_pos_logits(sim_mat, gt_prop)
        s_dist = F.softmax(new_logits, dim=0)
        t_dist = F.softmax(sim_mat/self.nav_t, dim=1)
        
        cost_mat = self.pairwise_cosine_dist(mu_s, f_t)
        source_loss = (self.s_par*cost_mat*s_dist).sum(0).mean() 
        target_loss = (((1-self.s_par)*cost_mat*t_dist).sum(1)*gt_prop.squeeze(1)).sum()
        # est_prop = s_dist.mean(1, keepdim=True)
        # log_gt_prop = (gt_prop + 1e-6).log()
        # log_est_prop = (est_prop + 1e-6).log()
        # kl_loss = (1-self.s_par)*(-torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop))
        
        loss = source_loss + target_loss
        return loss

# class EntropyLoss(nn.Module):
#     def __init__(self,nav_t,num_classes,device,weights=None):
#         super(EntropyLoss, self).__init__()
#         self.nav_t = nav_t
#         if weights is not None:
#             self.weights = weights
#         else:
#             self.weights = (torch.ones((num_classes,1))*(1/num_classes)).to(device)
            
#     def get_prob_logits(self, x, y):
#         x = F.normalize(x, p=2, dim=1)
#         y = F.normalize(y, p=2, dim=1)
#         return torch.matmul(x, y.T)
    
#     def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor) -> torch.Tensor:
#         prob_logits = self.get_prob_logits(mu_s,f_t)/self.nav_t
#         probs = F.softmax(prob_logits,dim=0)
#         return torch.sum(-self.weights * probs * torch.log(probs + 1e-6), dim=0).mean()

# class KLPropLoss(nn.Module):
#     """
#     CE between proportions
#     """
#     def __init__(self, ):
#         super(KLPropLoss, self).__init__()

#     def formawrd(self, probs: Tensor, target: Tensor) -> Tensor:
#         est_prop = probs.mean(dim=1, keepdim=True)
#         log_gt_prop = (target + 1e-6).log()
#         log_est_prop = (est_prop + 1e-6).log()
#         loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
#         return loss

# class Entropy_KLProp_Loss(nn.Module):

#     """
#     Simplify Implementaion of Entropy and KLProp Loss (MICCAI 2020)
#     Parameters:
#         - **nav_t** (float): temperature parameter (1 for all experiments)
#         - **beta** (float): learning rate/momentum update parameter for learning target proportions
#         - **num_classes** (int): total number of classes
#         - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

#     Inputs: mu_s, f_t
#         - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
#         - **f_t** (tensor): feature representations on target domain, :math:`f^t`

#     Shape:
#         - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

#     """

#     def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
#         super(Entropy_KLProp_Loss, self).__init__()
#         self.nav_t = nav_t
#         self.s_par = s_par
#         self.beta = beta
#         self.eps = 1e-6
         
#     def get_prob_logits(self, x, y):
#         x = F.normalize(x, p=2, dim=1)
#         y = F.normalize(y, p=2, dim=1)
#         return torch.matmul(x, y.T)


#     def forward(self, mu_s: torch.Tensor, f_t: torch.Tensor, gt_prop) -> torch.Tensor:
#         # Update proportions
#         prob_logits = self.get_prob_logits(mu_s,f_t)/self.nav_t
#         probs = F.softmax(prob_logits,dim=0)
#         est_prop = probs.mean(dim=1, keepdim=True)
        
#         log_gt_prop = (gt_prop + 1e-6).log()
#         log_est_prop = (est_prop + 1e-6).log()
        
#         weights = 1/gt_prop
#         weights = weights/torch.sum(weights)
        
#         # entropy_loss = torch.sum(-weights * probs * torch.log(probs + 1e-6), dim=0).mean()
#         entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=0).mean()
#         klprop_loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
#         loss = self.s_par*entropy_loss + (1-self.s_par)*klprop_loss
        
#         return loss

class Entropy_KLProp_Loss(nn.Module):

    """
    Simplify Implementaion of Entropy and KLProp Loss (MICCAI 2020)
    Parameters:
        - **nav_t** (float): temperature parameter (1 for all experiments)
        - **beta** (float): learning rate/momentum update parameter for learning target proportions
        - **num_classes** (int): total number of classes
        - **s_par** (float, optional): coefficient in front of the bi-directional loss. 0.5 corresponds to pct. 1 corresponds to using only t to mu. 0 corresponds to only using mu to t.

    Inputs: mu_s, f_t
        - **mu_s** (tensor): weight matrix of the linear classifier, :math:`mu^s`
        - **f_t** (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - mu_s: : math: `(K,F)`, f_t: :math:`(M, F)` where F means the dimension of input features.

    """

    def __init__(self, nav_t: float, beta: float, num_classes: int, device: torch.device, s_par: Optional[float] = 0.5, reduction: Optional[str] = 'mean'):
        super(Entropy_KLProp_Loss, self).__init__()
        self.nav_t = nav_t
        self.s_par = s_par
        self.beta = beta
        self.eps = 1e-6
         
    def forward(self, probs, gt_prop) -> torch.Tensor:
        # Update proportions
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        est_prop = probs.mean(dim=0, keepdim=True)
        log_gt_prop = (gt_prop + 1e-6).log()
        log_est_prop = (est_prop + 1e-6).log()
        
        
        # entropy_loss = torch.sum(-weights * probs * torch.log(probs + 1e-6), dim=0).mean()
        entropy_loss = torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
        klprop_loss = -torch.sum(est_prop * log_gt_prop) + torch.sum(est_prop * log_est_prop)
        loss = self.s_par*entropy_loss + (1-self.s_par)*klprop_loss
        
        return loss
    
class EntropyLoss(nn.Module):
    def __init__(self,num_classes,device,weights=None):
        super(EntropyLoss, self).__init__()
        if weights is not None:
            self.weights = weights
        else:
            self.weights = (torch.ones((1,num_classes))*(1/num_classes)).to(device)

    
    def forward(self, probs) -> torch.Tensor:
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        
        return torch.sum(-probs * torch.log(probs + 1e-6), dim=1).mean()
    
class EntropyClassMarginals(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs):
        avg_p = probs.mean(dim=[2, 3]) # avg along the pixels dim h x w -> size is batch x n_classes
        entropy_cm = torch.sum(avg_p * torch.log(avg_p + 1e-6), dim=1).mean()
        return entropy_cm
    
# class PseudoLabel_Loss(nn.Module):
#     def __init__(self):
#         super(PseudoLabel_Loss,self).__init__()
#         self.eps = 1e-10
    
#     def forward(self, pred, pseudo_label_teacher, drop_percent, prob_teacher):
#         batch_size, num_class, h, w = pred.shape
#         with torch.no_grad():
#             # drop pixels with high entropy
            
#             entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + self.eps), dim=1)

#             thresh = np.percentile(
#                 entropy[pseudo_label_teacher != 255].detach().cpu().numpy().flatten(), drop_percent
#             )
#             thresh_mask = entropy.ge(thresh).bool() * (pseudo_label_teacher != 255).bool()

#             pseudo_label_teacher[thresh_mask] = 255
#             weight = batch_size * h * w / torch.sum(pseudo_label_teacher != 255)

#         loss = weight * F.cross_entropy(pred, pseudo_label_teacher, ignore_index=255)  # [10, 321, 321]

#         return loss

    
class PseudoLabel_Loss(nn.Module):
    def __init__(self):
        super(PseudoLabel_Loss,self).__init__()
        self.eps = 1e-6
    
    def get_logits(self, prop):
        log_prior = torch.log(prop + self.eps)
        return log_prior
        
    def forward(self, pred, target, drop_percent, prob_teacher):
        # drop pixels with high entropy
        b, c, h, w  = pred.shape
        # neg_loss = 0
        # pdb.set_trace()
        with torch.no_grad():
            entropy = -torch.sum(prob_teacher * torch.log(prob_teacher + self.eps), dim=1)
            for i in range(c):
                if torch.sum(entropy[target == i]) > 10:

                    thresh = np.percentile(
                    entropy[target == i].detach().cpu().numpy().flatten(), drop_percent
                    )
                    thresh_mask = entropy.ge(thresh).bool() * (target == i).bool()
                    target[thresh_mask] = 255
                    # neg_prob = prob[:,i][thresh_mask]
                    # neg_target = torch.zeros_like(neg_prob).to(neg_prob.device)
                    # neg_target[neg_prob < 0.05] = 1
                    # neg_loss += -torch.mean(neg_target * self.get_logits(1-neg_prob))
        weight = b * h * w / torch.sum(target != 255)

        pos_loss = weight * F.cross_entropy(pred, target, ignore_index=255)  # [10, 321, 321]
        
        # loss = pos_loss + neg_loss

        return pos_loss

# class PseudoLabel_Loss(nn.Module):
#     def __init__(self):
#         super(PseudoLabel_Loss,self).__init__()
#         self.eps = 1e-6
    
#     def get_logits(self, prop):
#         log_prior = torch.log(prop + self.eps)
#         return log_prior
        
#     def forward(self, pred, prob, target, percent, entropy):
#         # drop pixels with high entropy
#         b, c, h, w  = pred.shape
#         neg_loss = 0
#         # pdb.set_trace()
#         for i in range(c):
            
#             thresh = np.percentile(
#             prob[:,i][target == i].detach().cpu().numpy().flatten(), percent
#         )
#             thresh_mask = prob[:,i].le(thresh).bool() * (target == i).bool()
#             target[thresh_mask] = 255
#             neg_prob = prob[:,i][thresh_mask]
#             neg_target = torch.zeros_like(neg_prob).to(neg_prob.device)
#             neg_target[neg_prob < 0.05] = 1
#             neg_loss += -torch.mean(neg_target * self.get_logits(1-neg_prob))

#         pos_loss = F.cross_entropy(pred, target, ignore_index=255)  # [10, 321, 321]
        
#         loss = pos_loss + neg_loss

#         return loss

class Curriculum_Style_Entropy_Loss(nn.Module):
    def __init__(self,alpha=0.002,gamma=2):
        super(Curriculum_Style_Entropy_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    
    def forward(self, probs) -> torch.Tensor:
        probs = rearrange(probs, 'b c h w -> (b h w) c')
        probs = F.softmax(probs,dim=1)
        entropy_map = torch.sum(-probs * torch.log(probs + 1e-6), dim=1)
        probs_hat = torch.mean(torch.exp(-3 * entropy_map).unsqueeze(dim=1) * probs, dim=0)
        loss_cel = self.alpha * ((1.7-entropy_map) ** self.gamma) * entropy_map
        loss_div = torch.sum(-probs_hat * torch.log(probs_hat + 1e-6))
        # pdb.set_trace()
        
        return loss_cel.mean()+loss_div


def intra_class_variance(prob, img):
    mean_std = torch.std(img * prob, dim=[2,3])
    return mean_std.mean()

def inter_class_variance(prob, img):
    mean_std = torch.std(torch.mean(img * prob, dim=[2,3]), dim=1)
    return mean_std.mean()

if __name__ == '__main__':
    criterion_proto = ProtoLoss(0.1,0.001,6,0,0.5)
    x = torch.randn(6, 256).to("cuda:0")
    y = torch.randn(150, 256).to("cuda:0")
    loss1, loss2 = criterion_proto(x, y)
    print(loss1)
