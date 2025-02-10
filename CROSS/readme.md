# CROSS: Feedback-Oriented Multi-Modal Dynamic Alignment in Recommendation Systems    
This repository is the official implementation of "CROSS: Feedback-Oriented Multi-Modal Dynamic Alignment in Recommendation Systems".   
It is an extension of our privious work FETTLE.  
# Framework
![Click here to view the framework](https://github.com/XMUDM/FETTLE/blob/main/CROSS/figs/framework.pdf)                

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
