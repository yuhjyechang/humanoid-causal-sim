import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from src.envs.humanoid.humanoid_causal_env import HumanoidCausalEnv

def main():
    n_envs = 16 # CPU mode, not 4096
    print(f"Initializing {n_envs} humanoids...")
    env = HumanoidCausalEnv(n_envs=n_envs)
    # use env's actual action dim
    action_dim = env.physics.action_dim
    print(f"Action dim detected: {action_dim}")

    obs, _ = env.reset()
    for i in range(200):
        actions = torch.randn(n_envs, action_dim).clamp(-1,1)
        obs, reward, done, truncated, info = env.step(actions)
        if i % 20 == 0:
            print(f"[{i}] avg height: {info['height']:.2f} | reward: {reward.mean():.2f}")
            if i % 100 == 0:
                print(f" -> Explain: {env.explain('fell')}")
                env.do(friction=0.1)

if __name__ == "__main__":
    main()
