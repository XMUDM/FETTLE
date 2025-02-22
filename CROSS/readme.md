# CROSS: Feedback-Oriented Multi-Modal Dynamic Alignment in Recommendation Systems    
This repository is the official implementation of "CROSS: Feedback-Oriented Multi-Modal Dynamic Alignment in Recommendation Systems".   
It is an extension of our privious work FETTLE.  
# Framework
![Framework](https://github.com/XMUDM/FETTLE/blob/main/CROSS/figs/framework.png)                

# Quickly used  
You only need to import the losses: CLALoss,ILALoss and CNALoss into the your code.  
e.g.
```python

self.ila_dt_loss = ILALoss(leaky_bi=True) # set leaky_bi=True --> IDLA
self.cla_loss = CLALoss(...)
self.cna_loss = CNALoss(...)

# <user, pos_item> are interacted user-item pairs.
ila_dt_loss = self.ila_dt_loss(u_emb, i_emb,  
                            v_emb, t_emb, user, pos_item)  

cla_loss = self.cla_loss(user_embeddings, item_embeddings,        
                            v_emb, t_emb, user, pos_item)  

cna_loss = self.cna_loss( item_embeddings, v_emb, t_emb, item)        

loss +=  self.iladt_weight * ila_dt_loss + self.cla_weight * cla_loss + self.cna_weight * cna_loss
```
# Dataset Description
- Baby/Sports/Clothing: These datasets are preprocessed by MMRec. Download from Google Drive: ![Baby/Sports/Elec][https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG] 
- Tiktok: The official website of the Tiktok dataset has been closed. MMSSL provided a preprocessed version of ![TikTok][https://github.com/HKUDS/MMSSL]. We use this version for our work.
