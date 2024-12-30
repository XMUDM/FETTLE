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

class ILADTLoss(nn.Module):
