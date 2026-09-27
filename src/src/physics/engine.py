
"""
Real Genesis backend - Unitree H1 humanoid, GPU-parallel, differentiable
Requires: pip install genesis-world torch
Genesis will auto-download H1 MJCF on first run
"""
import torch
import genesis as gs
from enum import Enum

class Backend(Enum):
    GENESIS = "genesis"
    MUJOCO = "mujoco"

class PhysicsEngine:
    def __init__(self, n_envs=4096, dt=0.002, substeps=10, backend=Backend.GENESIS):
        self.n_envs = n_envs
        self.backend = backend
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=substeps),
            viewer_options=gs.options.ViewerOptions(res=(1280,720), camera_fov=30),
            rigid_options=gs.options.RigidOptions(enable_collision=True),
        )
        # ground
        self.scene.add_entity(morph=gs.morphs.Plane())
        # H1 humanoid - Genesis includes MJCF assets, or use gs.morphs.MJCF(file="xml/unitree_h1/h1.xml")
        # If asset not found, it will fallback to simple humanoid URDF
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.MJCF(file="xml/unitree_h1/h1.xml"),
            )
        except Exception as e:
            print(f"[Physics] H1 MJCF not found locally ({e}), using URDF fallback")
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(file="urdf/humanoid/humanoid.urdf", fixedbase=False),
            )
        self.scene.build(n_envs=n_envs)
        self._dofs = self.robot.n_dofs
        print(f"[Physics] Built {n_envs} envs, dofs={self._dofs}")

    def reset(self, randomization=True):
        # randomize qpos slightly around standing pose
        qpos = self.robot.get_qpos()  # [n_envs, nq]
        if randomization:
            qpos += (torch.rand_like(qpos) - 0.5) * 0.05
        self.robot.set_qpos(qpos)
        self.robot.set_dofs_velocity(torch.zeros((self.n_envs, self._dofs), device=qpos.device))

    def step(self, actions: torch.Tensor):
        """
        actions: [n_envs, 21] in [-1,1] -> mapped to PD target
        """
        # PD control - Genesis handles it
        self.robot.control_dofs_position(actions)
        self.scene.step()

        # differentiable state
        qpos = self.robot.get_qpos()  # [n, nq]
        qvel = self.robot.get_dofs_velocity()  # [n, ndof]
        contacts = self.scene.rigid_solver.collider.get_contacts()  # dict
        pos = self.robot.get_pos()  # root pos
        return {
            "qpos": qpos,
            "qvel": qvel,
            "contacts": contacts,
            "root_pos": pos,
            "com": pos,  # simplified COM = root pos
        }

    def get_state(self):
        return {"qpos": self.robot.get_qpos()}

    def set_state(self, state_dict, is_intervention=False):
        if "friction" in state_dict:
            # intervention: change ground friction
            self.scene.rigid_solver.set_friction(state_dict["friction"])
