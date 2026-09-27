
import torch
import gymnasium as gym
import numpy as np
from src.physics.engine import PhysicsEngine, Backend
from src.causal.graph import CausalGraphManager
from src.causal.intervention_api import InterventionAPI

class HumanoidCausalEnv(gym.Env):
    """
    Humanoid standing / walking with WHY API
    Observation: qpos(27) + qvel(27) + root(7) + contact flag(4) = ~65, padded to 100
    Action: 21 DoF position targets
    """
    def __init__(self, n_envs=4096, backend=Backend.GENESIS):
        super().__init__()
        self.n_envs = n_envs
        self.physics = PhysicsEngine(n_envs=n_envs, backend=backend)
        self.causal = CausalGraphManager()
        self.intervention = InterventionAPI(self.physics, self.causal)

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(21,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)
        self.step_count = 0

    def reset(self, seed=None):
        self.physics.reset(randomization=True)
        self.step_count = 0
        return torch.zeros(self.n_envs, 100), {}

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float().cuda() if torch.cuda.is_available() else torch.from_numpy(actions).float()

        state = self.physics.step(actions)
        self.step_count += 1

        # --- Causal logging ---
        # Example: log if foot contact lost
        contacts = state.get("contacts", {})
        # In real impl, parse contacts to detect COM outside support polygon
        if self.step_count % 10 == 0:
            # dummy cause chain for now: torque -> contact -> COM
            self.causal.add_contact_event(
                t=self.step_count,
                body_a="foot_left",
                body_b="ground",
                force=float(torch.rand(1).item()),
                friction=0.8,
                cause_chain=["ankle_torque", "normal_force", "friction"]
            )

        # reward: stay upright + minimal energy
        root_pos = state["root_pos"]  # [n,3]
        height = root_pos[:, 2]
        reward = (height > 0.8).float()  # stay above 0.8m

        # done: fell
        done = height < 0.6

        # obs: pack qpos+qvel+root
        obs = torch.zeros(self.n_envs, 100, device=root_pos.device)
        obs[:, :3] = root_pos

        info = {
            "height": height.mean().item(),
            "contacts": str(contacts)[:200],
            "causal_graph": self.causal.export()
        }
        return obs, reward, done, False, info

    # --- WHY API ---
    def do(self, **kwargs):
        """Intervention: env.do(friction=0.1, pelvis_x=0.05)"""
        return self.intervention.do(**kwargs)

    def explain(self, failure="fell"):
        return self.causal.explain_failure(failure)

    def get_counterfactual(self, **do_ops):
        return self.intervention.counterfactual_rollout(**do_ops)
