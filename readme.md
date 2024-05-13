# Who To Align With: Feedback-Oriented Multi-Modal Alignment in Recommendation Systems
This repository is the official implementation of "Who To Align With: Feedback-Oriented Multi-Modal Alignment in Recommendation Systems".
Our paper is accepted by  SIGIR2024! 🎉

# Framework
![1712569859631](https://github.com/XMUDM/FETTLE/assets/77662961/8f802d8b-4a77-4b8d-a31b-32f28c16affc)

# quickly used
You only need to import the losses: CLALoss and ILALoss into the your code.
e.g.
```python
# <user, pos_item> are interacted user-item pairs.
ila_dt_loss = self.ila_dt_loss(u_emb, i_emb,
                            v_emb, t_emb, user, pos_item)

cla_loss = self.cla_loss(user_embeddings, item_embeddings,
                            v_emb, t_emb, user, pos_item)

loss +=  self.iladt_weight * ila_dt_loss + self.cla_weight * cla_loss
```



