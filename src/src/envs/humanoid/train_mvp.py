
"""
Train H1 to stand with 4096 parallel envs
Run: python -m src.envs.humanoid.train_mvp
"""
import torch
from src.envs.humanoid.humanoid_causal_env import HumanoidCausalEnv

def main():
    print("Initializing 4096 humanoids...")
    env = HumanoidCausalEnv(n_envs=4096)
    obs, _ = env.reset()

    for i in range(10000):
        # random action for MVP, replace with PPO later
        actions = torch.randn(4096, 21, device="cuda" if torch.cuda.is_available() else "cpu").clamp(-1,1)
        obs, reward, done, truncated, info = env.step(actions)

        if i % 100 == 0:
            print(f"[{i}] avg height: {info['height']:.2f} | reward: {reward.mean():.2f}")
            # probe causality every 500 steps
            if i % 500 == 0:
                cf = env.get_counterfactual(friction=0.0)
                expl = env.explain("fell")
                print(f"  -> Explain: {expl['cause_chain']} | Minimal fix: {expl['minimal_fix']}")
                print(f"  -> Counterfactual with friction=0: simulated")

if __name__ == "__main__":
    main()
