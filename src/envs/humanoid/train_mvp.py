"""
Train humanoid to stand/walk with causal supervision
"""
import torch
from src.envs.humanoid.humanoid_causal_env import HumanoidCausalEnv

def main():
    env = HumanoidCausalEnv(num_envs=4096)

    # Example RSL-RL / PPO loop
    for i in range(10000):
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)

        # Every 100 steps, probe causality
        if i % 100 == 0:
            cf = env.get_counterfactual(friction=0.0)
            expl = env.explain(failure="fell")
            print(f"[{i}] Counterfactual COM: {cf.get('com')}, Why fell: {expl['cause_chain']}")

if __name__ == "__main__":
    main()
