# Instead of just reward, add explanation loss
loss = ppo_loss + 0.3 * causal_loss
# causal_loss: can model predict contact force given do(friction=0)?
