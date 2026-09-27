# Humanoid Causal World Simulator
AAA presentation (Unreal Engine 5) + Causal Physics Brain for Humanoid AI

> Not just WHAT happened, but WHY it happened.

### Architecture
- **Renderer:** UE5.4+ (Nanite/Lumen/RTX) - Dumb renderer, USD-driven
- **Physics:** Genesis (GPU, differentiable, rigid+deformable) + MuJoCo (reference contact)
- **Causal Layer:** SCM graph, do() interventions, explain()
- **Env:** Gymnasium VecEnv, 4096 parallel humanoids

### Quick Start
```bash
pip install -r requirements.txt
python src/envs/humanoid/train_mvp.py --num_envs 1024
# For AAA view:
# Open UE5 project in /ue_project and enable USD Live Link
```

### The WHY API
```python
env = HumanoidCausalEnv()
obs, _ = env.reset()

obs, reward, done, info = env.step(action)

# Causal magic
counterfactual = env.do(mass_scale=2.0, friction=0.1)  # what if floor was slippery?
explanation = env.explain(failure="fell") 
# -> {'cause_chain': ['COM outside support polygon', 'ankle torque saturated'], 'minimal_fix': 'shift pelvis 3cm'}

graph = env.get_causal_graph() # NetworkX DiGraph of forces
```

See `docs/CAUSALITY.md` for training paradigm.
