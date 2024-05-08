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



