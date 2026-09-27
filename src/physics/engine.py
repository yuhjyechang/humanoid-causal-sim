"""
Dual backend: Genesis (primary, GPU, differentiable) + MuJoCo (ground truth contact)
"""
from enum import Enum
class Backend(Enum):
    GENESIS = "genesis"
    MUJOCO = "mujoco"

class PhysicsEngine:
    def __init__(self, backend=Backend.GENESIS, dt=0.002, substeps=2):
        self.backend = backend
        self.dt = dt
        self.substeps = substeps
        self._init_backend()

    def _init_backend(self):
        if self.backend == Backend.GENESIS:
            import genesis as gs
            gs.init(backend=gs.gpu)
            self.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=self.dt, substeps=self.substeps),
                viewer_options=gs.options.ViewerOptions(res=(1280, 720)),
            )
        else:
            import mujoco
            # load MJCF from assets/mjcf/humanoid.xml
            pass

    def step(self, actions):
        # returns: qpos, qvel, contact_forces, energies
        pass

    def get_state(self):
        # differentiable state dict
        return {}

    def set_state(self, state, is_intervention=False):
        # if is_intervention, marks node as do() in causal graph
        pass
