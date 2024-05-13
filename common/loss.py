# coding: utf-8
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kmeans_pytorch import kmeans as KMeans
from torch import cosine_similarity, Tensor
from torch.autograd import Function
from torch_scatter import scatter_add


class CLALoss(nn.Module):
    def __init__(self, K: int, D: int, gamma: float) -> None:
        """
        Args:
            K (int): Number of prototypes for each modality
            D (int): Dimension of each prototype
            gamma (float): Temperature parameter
        """
        super(CLALoss, self).__init__()
        self.K = K
        self.D = D
        self.gamma = gamma
        
        '''code implement details will be released after paper published'''


        self.gamma = nn.Parameter(torch.ones([]) * gamma)


    def sinkhorn(self, out):
        Q = torch.exp(out / 0.05).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        # dist.all_reduce(sum_Q)
        Q /= sum_Q
        # sinkhorn_iterations = 3
        sinkhorn_iterations = 3
        for it in range(sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            # dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, user_embeddings, id_embeddings, image_embeddings, text_embeddings, users, items,mode='SwAV',user_code=None,item_code=None,image_code=None,text_code=None):
        
        '''
        part code will be released after paper published
        '''

        with torch.no_grad():
            if mode=='SwAV':
                q_id = self.sinkhorn(code_id.detach())
                q_user = self.sinkhorn(code_user.detach())
                q_id_ii = self.sinkhorn(code_id_ii.detach())
                q_image_ii = self.sinkhorn(code_image_ii.detach())
                q_text_ii = self.sinkhorn(code_text_ii.detach())
            else:
                q_id = item_code.detach()
                q_user = user_code.detach()
                q_id_ii = q_id
                q_image_ii = image_code.detach()
                q_text_ii = text_code.detach()




        gamma = self.gamma
        if mode=='SwAV':
            loss = 0
            loss += -torch.mean(torch.sum(q_user * F.log_softmax(code_id / gamma, dim=1), dim=1))
            loss += -torch.mean(torch.sum(q_id * F.log_softmax(code_user / gamma, dim=1), dim=1))

            align_loss = 0
            align_loss += -torch.mean(torch.sum(q_id_ii * F.log_softmax(code_image_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_id_ii * F.log_softmax(code_text_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_image_ii * F.log_softmax(code_id_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_image_ii * F.log_softmax(code_text_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_text_ii * F.log_softmax(code_id_ii / gamma, dim=1), dim=1))
            align_loss += -torch.mean(torch.sum(q_text_ii * F.log_softmax(code_image_ii / gamma, dim=1), dim=1))
        else:
            loss = 0
            loss += F.cross_entropy(code_id[items], q_user[users])
            loss += F.cross_entropy(code_user[users], q_id[items])
            align_loss = 0
            align_loss += F.cross_entropy(code_image_ii[items], q_id_ii[items])
            align_loss += F.cross_entropy(code_text_ii[items], q_id_ii[items])
            align_loss += F.cross_entropy(code_id_ii[items], q_image_ii[items])
            align_loss += F.cross_entropy(code_text_ii[items], q_image_ii[items])
            align_loss += F.cross_entropy(code_id_ii[items], q_text_ii[items])
            align_loss += F.cross_entropy(code_image_ii[items], q_text_ii[items])

        return loss / 2 + align_loss / 6

class ILADTLoss(nn.Module):
    def __init__(self, dim=64, gamma=0.007):
        super(CLCRLossUltra, self).__init__()
        # self.logit_scale =  nn.Parameter(torch.ones([]) * np.log(1 / gamma))
        self.temp = nn.Parameter(gamma * torch.ones([]))
        '''code implement details will be released after paper published'''

    def compute_loss(self,user_embeddings,item_embeddings,image_embeddings,text_embeddings,user_id,item_id):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        ids = item_id.detach().cpu().numpy()
        '''
        part code will be released after paper published
        '''
        loss = 0
        dt_loss = 0

        '''
        loss compute details will be released after paper published
        '''
        return loss  + dt_loss

    def forward(self,user_embeddings,item_embeddings,image_embeddings,text_embeddings,user_id,item_id):
        return self.compute_loss(user_embeddings,item_embeddings,image_embeddings,text_embeddings,user_id,item_id)
