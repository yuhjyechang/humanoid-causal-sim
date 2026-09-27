# Why Causality Transforms Training

## Current paradigm: Correlation
Model learns: if pixels look like falling -> output corrective action.
Fails when lighting changes.

## New paradigm: Causal
Model learns graph: Action -> Joint Torque -> Contact Force -> COM -> Fall?
Trained on counterfactuals: "If friction were 0, would same action work? No."

## Training Losses
1. Task loss: PPO reward for walking
2. Causal consistency loss: Predict next contact given do(friction)
3. Explanation loss: LLM/VLM must output causal chain, supervised by graph

This gives you self-correcting humanoids: "I fell because ankle torque saturated, next time pre-shift COM."
