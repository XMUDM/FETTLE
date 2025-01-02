# coding: utf-8
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kmeans_pytorch import kmeans as KMeans
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
        self.feat2code = nn.Linear(D, K, bias=False)
        self.ii2code = nn.Linear(D, K, bias=False)



        self.gamma = nn.Parameter(torch.ones([]) * gamma)

    def get_refined_label(self,users,items,neg_items,user_embeddings,id_embeddings,text_embeddings,image_embeddings):
        with torch.no_grad():
            user_embeddings = F.normalize(user_embeddings[users],dim=1)
            id_embeddings = F.normalize(id_embeddings[items], dim=1)
            image_embeddings = F.normalize(image_embeddings[items], dim=1)
            text_embeddings = F.normalize(text_embeddings[items], dim=1)
            id_embeddings_neg = F.normalize(id_embeddings[neg_items], dim=1)
            image_embeddings_neg = F.normalize(image_embeddings[neg_items], dim=1)
            text_embeddings_neg = F.normalize(text_embeddings[neg_items], dim=1)

            code_user= user_embeddings  @ self.feat2code.weight.t()
            code_item = id_embeddings @ self.feat2code.weight.t()
            code_item_neg = id_embeddings_neg @ self.feat2code.weight.t()
            code_id_ii = id_embeddings @ self.ii2code.weight.t()
            code_id_ii_neg = id_embeddings_neg @ self.ii2code.weight.t()
            code_image_ii = image_embeddings @ self.ii2code.weight.t()
            code_image_ii_neg = image_embeddings_neg @ self.ii2code.weight.t()
            code_text_ii = text_embeddings @ self.ii2code.weight.t()
            code_text_ii_neg = text_embeddings_neg @ self.ii2code.weight.t()
            alpha = (code_id_ii+code_image_ii+code_text_ii)/3

            alpha_neg =  (code_id_ii_neg+code_image_ii_neg+code_text_ii_neg)/3
            y_hat = alpha*torch.sum(code_user*code_item,dim=1)-alpha_neg*torch.sum(code_user*code_item_neg,dim=1)
            y_hat =( y_hat+torch.ones_like(y_hat)) /2
            # 计算id2image的差异性
        return y_hat.detach()


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

    def kmeans(self,out):

        cluster_labels,_ = KMeans(X=out, num_clusters=self.K, distance='euclidean', device=out.device).long().to(out.device)
        # 对cluster_centers_重排，从0开始


        # cluster_centers = torch.from_numpy(cluster_labels).to(out.device)
        # 对cluster_centers_重排，从0开始
        # _, cluster_centers = torch.unique(cluster_centers, sorted=True, return_inverse=True, dim=0)

        return cluster_labels

    def forward(self, user_embeddings, id_embeddings, image_embeddings, text_embeddings, users, items,mode='SwAV',user_code=None,item_code=None,image_code=None,text_code=None):
        with torch.no_grad():
            self.gamma.clamp_(0.01, 0.99)
        # 归一化

        user_embeddings = F.normalize(user_embeddings[users], dim=1)
        id_embeddings = F.normalize(id_embeddings[items], dim=1)
        image_embeddings = F.normalize(image_embeddings[items], dim=1)
        text_embeddings = F.normalize(text_embeddings[items], dim=1)

        with torch.no_grad():
            w = self.feat2code.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.feat2code.weight.copy_(w)
            w = self.ii2code.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.ii2code.weight.copy_(w)

        code_user = user_embeddings @ self.feat2code.weight.t()
        code_id = id_embeddings @ self.feat2code.weight.t()
        code_id_ii = id_embeddings @ self.ii2code.weight.t()
        code_image_ii = image_embeddings @ self.ii2code.weight.t()
        code_text_ii = text_embeddings @ self.ii2code.weight.t()


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

def normalize(embeddings):
    mean = embeddings.mean(dim=0)
    std = embeddings.std(dim=0)
    return (embeddings - mean) / (std+1e-6)

def InfoNCE(view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        
        return torch.mean(cl_loss)

class ILALoss(nn.Module):
    def __init__(self, dim=64, gamma=0.007,leaky_bi=False):
        super(ILALoss, self).__init__()
        # self.logit_scale =  nn.Parameter(torch.ones([]) * np.log(1 / gamma))
        self.temp = nn.Parameter(gamma * torch.ones([]))
        self.i2t_map = nn.Linear(dim, dim, bias=False)
        self.t2i_map = nn.Linear(dim, dim, bias=False)
        self.i2d_map = nn.Linear(dim, dim, bias=False)
        self.d2i_map = nn.Linear(dim, dim, bias=False)
        self.t2d_map = nn.Linear(dim, dim, bias=False)
        self.d2t_map = nn.Linear(dim, dim, bias=False)
        self.leaky_bi = leaky_bi

    def forward(self,user_embeddings,item_embeddings,image_embeddings,text_embeddings,user_id,item_id,epoch_idx=None):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        item_id,indice = torch.sort(item_id)
        unique_items,remap_indexs, counts = torch.unique(item_id,return_inverse=True,return_counts=True,sorted=True)
        user_id = user_id[indice]

        user_embeddings = user_embeddings / user_embeddings.norm(dim=1, keepdim=True)
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)

        uid_scores = torch.sum(user_embeddings[user_id] *
                               item_embeddings[item_id], dim=1)
        uii_scores = torch.sum(user_embeddings[user_id] *
                                 image_embeddings[item_id], dim=1)
        uit_scores = torch.sum(user_embeddings[user_id] *
                                    text_embeddings[item_id], dim=1)

  
        with torch.no_grad():
            uid_scores = scatter_add(uid_scores / counts[remap_indexs], remap_indexs, dim=0)
            uii_scores = scatter_add(uii_scores / counts[remap_indexs], remap_indexs, dim=0)
            uit_scores = scatter_add(uit_scores / counts[remap_indexs], remap_indexs, dim=0)


            if self.leaky_bi:
                t_i_mask = uii_scores + torch.var(uii_scores)* (torch.exp(1-uii_scores)-1) > uit_scores
                i_t_mask = uit_scores +  torch.var(uit_scores)* (torch.exp(1-uit_scores)-1)> uii_scores
                i_d_mask = uid_scores +  torch.var(uid_scores)* (torch.exp(1-uid_scores)-1)> uii_scores
                d_i_mask = uii_scores +  torch.var(uii_scores)* (torch.exp(1-uii_scores)-1)> uid_scores
                t_d_mask = uid_scores +  torch.var(uid_scores)* (torch.exp(1-uid_scores)-1)> uit_scores
                d_t_mask = uit_scores +  torch.var(uit_scores)* (torch.exp(1-uit_scores)-1)> uid_scores
            else:
                t_i_mask = uii_scores > uit_scores
                i_t_mask = uit_scores > uii_scores
                i_d_mask = uid_scores > uii_scores
                d_i_mask = uii_scores > uid_scores
                t_d_mask = uid_scores > uit_scores
                d_t_mask = uit_scores > uid_scores

        item_embeddings = item_embeddings[unique_items] # 7k -> unique items
        image_embeddings = image_embeddings[unique_items]
        text_embeddings = text_embeddings[unique_items]

        image_features_norm = image_embeddings
        text_features_norm = text_embeddings
        cf_features_norm = item_embeddings

        image_features_i2t = self.i2t_map(image_embeddings) + image_embeddings
        image_features_i2d = self.i2d_map(image_embeddings) + image_embeddings
        text_features_t2i = self.t2i_map(text_embeddings) + text_embeddings
        text_features_t2d = self.t2d_map(text_embeddings) + text_embeddings
        cf_features_d2i = self.d2i_map(item_embeddings) + item_embeddings
        cf_features_d2t = self.d2t_map(item_embeddings) + item_embeddings


        image_features_norm_i2t = image_features_i2t / image_features_i2t.norm(dim=1, keepdim=True)
        image_features_norm_i2d = image_features_i2d / image_features_i2d.norm(dim=1, keepdim=True)
        text_features_norm_t2i = text_features_t2i / text_features_t2i.norm(dim=1, keepdim=True)
        text_features_norm_t2d = text_features_t2d / text_features_t2d.norm(dim=1, keepdim=True)
        cf_features_norm_d2i = cf_features_d2i / cf_features_d2i.norm(dim=1, keepdim=True)
        cf_features_norm_d2t = cf_features_d2t / cf_features_d2t.norm(dim=1, keepdim=True)


        logits_image_cf = (image_features_norm_i2d @ cf_features_norm.t().detach())[i_d_mask] / self.temp
        logits_cf_image =  (cf_features_norm_d2i @ image_features_norm.t().detach())[d_i_mask] / self.temp
        logits_cf_text = (cf_features_norm_d2t @ text_features_norm.t().detach())[d_t_mask]  / self.temp
        logits_text_cf = (text_features_norm_t2d @ cf_features_norm.t().detach())[t_d_mask] / self.temp
        logits_image_text = (image_features_norm_i2t @ text_features_norm.detach().t())[i_t_mask]  / self.temp
        logits_text_image =  (text_features_norm_t2i @ image_features_norm.detach().t())[t_i_mask]  / self.temp

        # 判断是否存在nan

        loss = 0


        labels = torch.arange(unique_items.shape[0]).to(logits_image_cf.device)

        if t_i_mask.sum() > 0:
            t2i_loss = F.cross_entropy(logits_text_image,labels[t_i_mask], reduction='sum')
            loss+=t2i_loss
        if i_t_mask.sum() > 0:
            i2t_loss = F.cross_entropy(logits_image_text,labels[i_t_mask], reduction='sum')
            loss += i2t_loss
        if i_d_mask.sum() > 0:
            i2d_loss = F.cross_entropy(logits_image_cf,labels[i_d_mask], reduction='sum')
            loss += i2d_loss
        if d_i_mask.sum() > 0:
            d2i_loss = F.cross_entropy(logits_cf_image,labels[d_i_mask], reduction='sum')
            loss +=d2i_loss
        if t_d_mask.sum() > 0:
            t2d_loss = F.cross_entropy(logits_text_cf,labels[t_d_mask], reduction='sum')
            loss += t2d_loss
        if d_t_mask.sum() > 0:
            d2t_loss = F.cross_entropy(logits_cf_text,labels[d_t_mask], reduction='sum')
            loss += d2t_loss
        loss = loss / (unique_items.shape[0])

        loss_align = 0
        user_embeddings = user_embeddings.detach()
        if i_t_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * image_features_norm_i2t[remap_indexs],dim=1)
            pos_score = scatter_add(pos_score/counts[remap_indexs],remap_indexs,dim=0)
            loss_align += -torch.mean(pos_score[i_t_mask] - uii_scores[i_t_mask])
        if t_i_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * text_features_norm_t2i[remap_indexs],dim=1)
            pos_score = scatter_add(pos_score/counts[remap_indexs],remap_indexs,dim=0)
            loss_align += -torch.mean(pos_score[t_i_mask] - uit_scores[t_i_mask])
        if i_d_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * image_features_norm_i2d[remap_indexs],dim=1)
            pos_score = scatter_add(pos_score/counts[remap_indexs],remap_indexs,dim=0)
            loss_align += -torch.mean(pos_score[i_d_mask] - uii_scores[i_d_mask])
        if d_i_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * cf_features_norm_d2i[remap_indexs],dim=1)
            pos_score = scatter_add(pos_score/counts[remap_indexs],remap_indexs,dim=0)
            loss_align += -torch.mean(pos_score[d_i_mask] - uid_scores[d_i_mask])
        if t_d_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * text_features_norm_t2d[remap_indexs],dim=1)
            pos_score = scatter_add(pos_score/counts[remap_indexs],remap_indexs,dim=0)
            loss_align += -torch.mean(pos_score[t_d_mask] - uit_scores[t_d_mask])
        if d_t_mask.sum() > 0:
            pos_score = torch.sum(user_embeddings[user_id] * cf_features_norm_d2t[remap_indexs],dim=1)
            pos_score = scatter_add(pos_score/counts[remap_indexs],remap_indexs,dim=0)
            loss_align += -torch.mean(pos_score[d_t_mask] - uid_scores[d_t_mask])
        loss_align = loss_align / 6

        return loss +  loss_align

class CNALoss(nn.Module):
    def __init__(self,device,dim=64,cl_temp=0.2,topk=5,reduction='mean', dataset='baby',verbose=False):
        super(CNALoss, self).__init__()
        self.item_best_friends = torch.load(f'/your dir/{dataset}/{dataset}_item_top_{topk}.pt').to(device)
        self.dataset = dataset
        self.reduction = reduction
        if reduction == 'mean':
            self.weight = nn.Parameter(torch.ones(topk),requires_grad=False)
        elif reduction == 'param_attn':
            self.weight =  nn.Parameter(torch.ones(topk),requires_grad=True)
        else:
            raise ValueError('reduction should be mean or param_attn')
            
        self.topk = topk
        self.cl_temp = cl_temp
    
    def sq_sum(self, emb):
        return 1. / 2 * (emb ** 2).sum()
    
    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)
    
    def forward(self,item_embeddings,item_image_embeddings,item_text_embeddings,item_id,epoch_idx=None):
        item_id,indice = torch.sort(item_id)
        unique_items, remap_indices, counts = torch.unique(item_id, return_inverse=True, return_counts=True, sorted=True)

        # 标准化嵌入向量 shape: [batch_size, dim]
        item_embeddings = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)
        item_image_embeddings = item_image_embeddings / item_image_embeddings.norm(dim=1, keepdim=True)
        item_text_embeddings = item_text_embeddings / item_text_embeddings.norm(dim=1, keepdim=True)
        
        # 朋友嵌入向量 shape: [batch_size, topk, dim]
        item_friends_embeddings = item_embeddings[self.item_best_friends[unique_items]]
        item_friends_image_embeddings = item_image_embeddings[self.item_best_friends[unique_items]]
        item_friends_text_embeddings = item_text_embeddings[self.item_best_friends[unique_items]]

        # id_weight shape: [batch_size, topk,1]

        weight = F.softmax(self.weight,dim=0).view(-1,self.topk,1)
        item_friends_embeddings = torch.sum(weight * item_friends_embeddings,dim=1)
        item_friends_image_embeddings = torch.sum(weight * item_friends_image_embeddings,dim=1)
        item_friends_text_embeddings = torch.sum(weight * item_friends_text_embeddings,dim=1)
        
        cl_loss = sum(
            self.InfoNCE(embeddings, friend_embeddings, self.cl_temp)
            for embeddings, friend_embeddings in [
                (item_embeddings[unique_items], item_friends_embeddings),
                (item_image_embeddings[unique_items], item_friends_image_embeddings),
                (item_text_embeddings[unique_items], item_friends_text_embeddings),
            ]
        ) / 3

        reg_loss = self.sq_sum(item_friends_embeddings) + self.sq_sum(item_friends_image_embeddings) + self.sq_sum(item_friends_text_embeddings)
        return cl_loss, reg_loss
