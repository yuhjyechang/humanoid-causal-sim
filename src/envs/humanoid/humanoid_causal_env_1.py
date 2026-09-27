import gymnasium as gym
import numpy as np
from src.physics.engine import PhysicsEngine, Backend
from src.causal.graph import CausalGraphManager
from src.causal.intervention_api import InterventionAPI

class HumanoidCausalEnv(gym.Env):
    """
    MVP: Unitree H1 / Fourier GR1 style humanoid
    Observation: proprio + COM + contact + causal embedding
    Action: 19-21 DoF joint positions
    """
    def __init__(self, num_envs=4096, backend=Backend.GENESIS, render_mode="usd"):
        super().__init__()
        self.num_envs = num_envs
        self.physics = PhysicsEngine(backend=backend)
        self.causal_graph = CausalGraphManager()
        self.intervention = InterventionAPI(self.physics, self.causal_graph)

        # 21 DoF humanoid: 6 legs + 5 per arm + 3 torso + 1 head
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(21,))
        # obs: qpos(27) + qvel(27) + contact(14) + causal_feat(32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,))

    def reset(self, seed=None):
        # randomize: mass, friction, COM offset, push disturbances
        # log randomization to causal graph
        return np.zeros(100), {}

    def step(self, action):
        # 1. Physics step
        state = self.physics.step(action)
        # 2. Update causal graph with contacts
        # self.causal_graph.add_contact_event(...)
        # 3. Compute reward: standing + energy efficiency + balance
        reward = 0.0
        done = False
        info = {
            'contact_forces': state.get('contact', {}),
            'com': state.get('com'),
            'causal_graph': self.causal_graph.export()
        }
        return np.zeros(100), reward, done, False, info

    # --- THE WHY API ---
    def do(self, **kwargs):
        """Intervention: env.do(friction=0.1)"""
        return self.intervention.do(**kwargs)

    def explain(self, failure="fell"):
        return self.causal_graph.explain_failure(failure)

    def get_causal_graph(self):
        return self.causal_graph.graph

    def get_counterfactual(self, **do_ops):
        return self.intervention.counterfactual_rollout(**do_ops)
