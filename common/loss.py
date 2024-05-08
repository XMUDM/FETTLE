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

    def kmeans(self,out):

        cluster_labels,_ = KMeans(X=out, num_clusters=self.K, distance='euclidean', device=out.device).long().to(out.device)
        # 对cluster_centers_重排，从0开始


        # cluster_centers = torch.from_numpy(cluster_labels).to(out.device)
        # 对cluster_centers_重排，从0开始
        # _, cluster_centers = torch.unique(cluster_centers, sorted=True, return_inverse=True, dim=0)

        return cluster_labels

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

    def __init__(self, gamma=0.007,dim=64):
        super(CLCRLossSE, self).__init__()
        self.logit_scale =  nn.Parameter(torch.ones([]) * np.log(1 / gamma))
        # self.mapper = nn.Linear(dim,dim, bias=False)

        self.i2t_map = nn.Linear(dim, dim, bias=False)
        self.t2i_map = nn.Linear(dim, dim, bias=False)
        self.i2d_map = nn.Linear(dim, dim, bias=False)
        self.d2i_map = nn.Linear(dim, dim, bias=False)
        self.t2d_map = nn.Linear(dim, dim, bias=False)
        self.d2t_map = nn.Linear(dim, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros([]))
    def bpr_loss(self, users, pos_items, neg_items):
        # print("bpr loss shape:", users.shape, pos_items.shape, neg_items.shape)
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        return mf_loss


    def add_noise(self,user_embeddings,item_embeddings,noise_rate):
        if noise_rate == 0:
            return item_embeddings
        # Calculate the sampling probabilities
        add_noise_num = int(item_embeddings.shape[0] * noise_rate)
        sim_scores = torch.sum(F.normalize(user_embeddings) * F.normalize(item_embeddings), dim=-1)
        sampling_probs = torch.exp(sim_scores)
        sampling_probs /= torch.sum(sampling_probs)

        # Sample indices based on the probabilities
        neg_indices = torch.multinomial(sampling_probs, num_samples=add_noise_num, replacement=False)
        # Add noise
        mask = torch.zeros(item_embeddings.shape[0]).to(item_embeddings.device)
        mask[neg_indices] = 1
        # print(torch.randn_like(item_embeddings).shape)
        noise_embeddings = mask.view(-1,1) * torch.normal(0,0.01,item_embeddings.shape).to(item_embeddings.device)
        return item_embeddings+noise_embeddings

    def add_noise_align(self, item_embeddings,noise_rate):
        if noise_rate == 0:
            return item_embeddings
        # Calculate the sampling probabilities
        add_noise_num = int(item_embeddings.shape[0] * noise_rate)
        # Sample indices based on the probabilities
        neg_indices = torch.multinomial(torch.ones(item_embeddings.shape[0]).to(item_embeddings.device), num_samples=add_noise_num, replacement=False)
        # Add noise
        mask = torch.zeros(item_embeddings.shape[0]).to(item_embeddings.device)
        mask[neg_indices] = 1
        # print(torch.randn_like(item_embeddings).shape)
        noise_embeddings = mask.view(-1,1) * torch.normal(0,0.01,item_embeddings.shape).to(item_embeddings.device)
        return item_embeddings+noise_embeddings

    def forward(self,user_embeddings,item_embeddings,image_embeddings,text_embeddings,user_id,item_id, neg_item_id=None):
        # 如果一个模态的得分均值很大，因此这是一个优势模态，相反如果一个模态它的得分均值很小，那么这是一个劣势模态
        # 劣势模态需要向优势模态看齐，因此需要将劣势模态向优势模态靠拢
        # 判断embeddings中是否有NAN
        # print("shape:", user_embeddings[user_id].shape, text_embeddings[item_id].shape)
        # mf_t_loss = self.bpr_loss(user_embeddings[user_id], text_embeddings[item_id], text_embeddings[neg_item_id])
        # mf_v_loss = self.bpr_loss(user_embeddings[user_id], image_embeddings[item_id], image_embeddings[neg_item_id])
        # mf_cf_loss = self.bpr_loss(user_embeddings[user_id], item_embeddings[item_id], item_embeddings[neg_item_id])
        # s_id = 1 - mf_t_loss/(mf_t_loss+mf_v_loss+mf_cf_loss)
        # s_ii = 1 - mf_v_loss/(mf_t_loss+mf_v_loss+mf_cf_loss)
        # s_it = 1 - mf_cf_loss/(mf_t_loss+mf_v_loss+mf_cf_loss)

        item_id,indice = torch.sort(item_id) 
        unique_items,remap_indexs, counts = torch.unique(item_id,return_inverse=True,return_counts=True,sorted=True)
    
        # 重新计算得分：给定用户序列： [U,E]，商品序列[I,E], 通过交互序列构建交互矩阵[U,I],使用两个序列作矩阵乘法，得到得分矩阵[U,I]， 进行归一化后计算与交互矩阵的相似度
        # 计算得分：用三个模态：id, image, text分别计算BPR Loss，取最小的作为优势模态，最大的作为劣势模态，加噪比与Loss大小成反比
        unique_users,remap_user_indexs, user_counts = torch.unique(user_id,return_inverse=True,return_counts=True,sorted=True)

        # # remap loss

        inter_matrix = torch.zeros((unique_users.shape[0],unique_items.shape[0])).to(user_embeddings.device)
        inter_matrix[remap_user_indexs,remap_indexs] = 1
        # print("inter matrix shape:", inter_matrix.shape)
        # print("user embeddings shape:", user_embeddings.shape)
        user_seq = user_embeddings[unique_users]
        item_seq = item_embeddings[unique_items]
        item_text_seq = text_embeddings[unique_items]
        item_image_seq = image_embeddings[unique_items]
        # print("unique user range:", unique_users[0],unique_users[-1] ,"unique item range:", unique_items[0], unique_items[-1])
        # print("shape:", user_seq.shape, item_seq.shape)
        uid_score = F.normalize(user_seq) @ F.normalize(item_seq).t()
        uii_score = F.normalize(user_seq) @ F.normalize(item_image_seq).t()
        uit_score = F.normalize(user_seq) @ F.normalize(item_text_seq).t()
        # 计算得分矩阵与交互矩阵的相似度
        # 使用交互矩阵作为mask，只计算交互过的商品的得分
        uid_score = uid_score * inter_matrix
        uii_score = uii_score * inter_matrix
        uit_score = uit_score * inter_matrix
        uid_scores = cosine_similarity(uid_score,inter_matrix)
        uii_scores = cosine_similarity(uii_score,inter_matrix)
        uit_scores= cosine_similarity(uit_score,inter_matrix)

        s_id = torch.mean(uid_scores) 
        s_ii = torch.mean(uii_scores)
        s_it = torch.mean(uit_scores)


        modals_score = torch.softmax(torch.tensor([s_id,s_ii,s_it]),dim=0) - 1/3    
        # print(modals_score)
        noise_rate = torch.clamp((1/(1+ np.exp(-100 * (modals_score))) - 1/2),min=1e-3)
        # print(noise_rate)
        noised_item_embeddings = self.add_noise(user_embeddings[user_id[indice]],item_embeddings[item_id],noise_rate[0])
      
        noised_image_embeddings = self.add_noise(user_embeddings[user_id[indice]],image_embeddings[item_id],noise_rate[1])
        noised_text_embeddings = self.add_noise(user_embeddings[user_id[indice]],text_embeddings[item_id],noise_rate[2])
        uid_scores = torch.sum(F.normalize(user_embeddings)[user_id[indice]] *
                               F.normalize(noised_item_embeddings), dim=1)
        uii_scores = torch.sum(F.normalize(user_embeddings)[user_id[indice]] *
                                 F.normalize(noised_image_embeddings), dim=1)
        uit_scores = torch.sum(F.normalize(user_embeddings)[user_id[indice]] *
                                    F.normalize(noised_text_embeddings), dim=1)

        uid_scores = torch.bincount(remap_indexs,weights=uid_scores/counts[remap_indexs]) # 重合的商品--平均
        uii_scores = torch.bincount(remap_indexs,weights=uii_scores/counts[remap_indexs])
        uit_scores = torch.bincount(remap_indexs,weights=uit_scores/counts[remap_indexs])

        # print("before", item_embeddings.shape, noised_item_embeddings.shape)
        item_embeddings = item_embeddings[unique_items] # 7k -> unique items
        image_embeddings = image_embeddings[unique_items]
        text_embeddings = text_embeddings[unique_items]
    

        # print('ti_mask',t_i_mask.shape)
        t_i_mask = uii_scores > uit_scores
        i_t_mask = uit_scores > uii_scores
        i_d_mask = uid_scores > uii_scores
        d_i_mask = uii_scores > uid_scores
        t_d_mask = uid_scores > uit_scores
        d_t_mask = uit_scores > uid_scores

        image_features_norm = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)
        text_features_norm = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)
        cf_features_norm = item_embeddings / item_embeddings.norm(dim=1, keepdim=True)
       
        image_features_norm_i2t = self.i2t_map(image_features_norm) + image_features_norm
        image_features_norm_i2d = self.i2d_map(image_features_norm) + image_features_norm
        text_features_norm_t2i = self.t2i_map(text_features_norm) + text_features_norm
        text_features_norm_t2d = self.t2d_map(text_features_norm) + text_features_norm
        cf_features_norm_d2i = self.d2i_map(cf_features_norm) + cf_features_norm
        cf_features_norm_d2t = self.d2t_map(cf_features_norm) + cf_features_norm
        # print("shape", image_features_norm.shape, noised_image_features_norm.shape)

        logit_scale = self.logit_scale.exp() #放缩logits
        # add noise to cf_features_norm
        # noised_cf_features_norm = self.add_noise_align(cf_features_norm,noise_rate[0])


        logits_image_cf = logit_scale  * (image_features_norm_i2d @ cf_features_norm.t().detach())
        # len[unique_items]*len[unique_items]
        logits_cf_image = logit_scale * (cf_features_norm_d2i @ image_features_norm.t().detach())
    
        logits_cf_text = logit_scale  * (cf_features_norm_d2t @ text_features_norm.t().detach())
        logits_text_cf = logit_scale  * (text_features_norm_t2d @ cf_features_norm.t().detach())
        # logits_image_text = logits_text_image.t()
        logits_image_text = logit_scale * (image_features_norm_i2t @ text_features_norm.detach().t())
        logits_text_image = logit_scale * (text_features_norm_t2i @ image_features_norm.detach().t())

        # 判断是否存在nan

        loss = 0
        
        labels = torch.arange(len(unique_items)).to(logits_image_cf.device)
        # print("logits,", logits_cf_image.shape, logits_cf_image.device)
        # print("mask,", t_i_mask.shape, "labels shape", labels.shape)
        # print("labels:", labels)

        if t_i_mask.sum() > 0:
            # 假设： 5个样本， 1、3、5 -> t_i mask: 
            # print("logits shape:",logits_text_image.shape, "labels shape", labels.shape) # i2t: 500个 neg： 499 ｜ 非i2t 
            loss += F.cross_entropy(logits_text_image[t_i_mask],labels[t_i_mask], reduction='sum')
            # print("test", logits_text_image[t_i_mask].shape, labels[t_i_mask].shape)
        if i_t_mask.sum() > 0:
            loss += F.cross_entropy(logits_image_text[i_t_mask],labels[i_t_mask], reduction='sum')
        if i_d_mask.sum() > 0:
            loss += 2*F.cross_entropy(logits_image_cf[i_d_mask],labels[i_d_mask], reduction='sum')
        if d_i_mask.sum() > 0:
            loss += F.cross_entropy(logits_cf_image[d_i_mask],labels[d_i_mask], reduction='sum')
        if t_d_mask.sum() > 0:
            loss += 2*F.cross_entropy(logits_text_cf[t_d_mask],labels[t_d_mask], reduction='sum')
        if d_t_mask.sum() > 0:
            loss += F.cross_entropy(logits_cf_text[d_t_mask],labels[d_t_mask], reduction='sum')

        return loss / unique_items.shape[0]


    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, *embeddings):
        l2_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            l2_loss += torch.sum(embedding**2)*0.5
        return l2_loss