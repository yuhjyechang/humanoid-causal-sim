import torch
from enum import Enum

class Backend(Enum):
    GENESIS = "genesis"
    MUJOCO = "mujoco"

class PhysicsEngine:
    def __init__(self, n_envs=16, dt=0.002, substeps=10, backend=Backend.GENESIS):
        self.n_envs = n_envs
        self._dofs = 6 # will be overwritten
        self.mode = "dummy"
        try:
            import genesis as gs
            try:
                print("[Physics] Trying GPU...")
                gs.init(backend=gs.gpu)
                self.mode = "genesis_gpu"
            except Exception as e:
                print(f"[Physics] GPU failed ({e}), using CPU")
                gs.init(backend=gs.cpu)
                self.mode = "genesis_cpu"

            self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt, substeps=substeps))
            self.scene.add_entity(morph=gs.morphs.Plane())
            try:
                self.robot = self.scene.add_entity(gs.morphs.MJCF(file="xml/unitree_h1/h1.xml"))
            except:
                print("[Physics] H1 not found, using Box (6 DOF)")
                self.robot = self.scene.add_entity(gs.morphs.Box(size=(0.5,0.3,1.7)))

            self.scene.build(n_envs=n_envs)
            self._dofs = self.robot.n_dofs if hasattr(self.robot, 'n_dofs') else self.robot.n_qs
            print(f"[Physics] Built {n_envs} envs, DOFs={self._dofs}, mode={self.mode}")

        except Exception as e:
            print(f"[Physics] Dummy fallback: {e}")
            self.mode = "dummy"
            self._height = torch.ones(n_envs) * 1.7

    @property
    def action_dim(self):
        return self._dofs

    def reset(self, randomization=True):
        if self.mode.startswith("genesis"):
            try:
                qpos = self.robot.get_qpos()
                if randomization:
                    qpos += (torch.rand_like(qpos) - 0.5) * 0.05
                self.robot.set_qpos(qpos)
            except: pass

    def step(self, actions: torch.Tensor):
        # auto-resize actions to match robot DOF
        if actions.shape[1]!= self._dofs:
            # if H1 (21) but Box (6), take first 6 or repeat
            if actions.shape[1] > self._dofs:
                actions = actions[:, :self._dofs]
            else:
                # pad
                pad = torch.zeros(actions.shape[0], self._dofs - actions.shape[1], device=actions.device)
                actions = torch.cat([actions, pad], dim=1)

        if self.mode == "dummy":
            self._height -= torch.rand(self.n_envs) * 0.01
            self._height = torch.clamp(self._height, 0.3, 1.8)
            h = self._height
            root = torch.stack([torch.zeros_like(h), torch.zeros_like(h), h], dim=1)
            return {"qpos": torch.zeros(self.n_envs, 27), "qvel": torch.zeros(self.n_envs, 27),
                    "contacts": {}, "root_pos": root, "com": root}

        self.robot.control_dofs_position(actions)
        self.scene.step()
        return {"qpos": self.robot.get_qpos(), "qvel": self.robot.get_dofs_velocity(),
                "contacts": {}, "root_pos": self.robot.get_pos(), "com": self.robot.get_pos()}

    def get_state(self): return {"dummy": 1}
    def set_state(self, state_dict, is_intervention=False):
        print(f"[Intervention] do({state_dict})")
