from .graph import CausalGraphManager

class InterventionAPI:
    """Exposed to env: env.do(), env.explain()"""
    def __init__(self, physics_engine, graph_manager: CausalGraphManager):
        self.physics = physics_engine
        self.graph = graph_manager

    def do(self, **kwargs):
        """e.g. env.do(friction=0.0, pelvis_mass=1.5)"""
        # 1. Mutilate graph
        for k, v in kwargs.items():
            self.graph.do(k, v)
        # 2. Re-simulate forward from intervention point
        self.physics.set_state(kwargs, is_intervention=True)
        return self.physics.get_state()

    def counterfactual_rollout(self, steps=100, **do_ops):
        """Would the humanoid have fallen if ...?"""
        # save state, apply do, rollout, restore
        pass
