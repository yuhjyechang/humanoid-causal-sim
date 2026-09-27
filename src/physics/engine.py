import genesis as gs
import torch

class PhysicsEngine:
    def __init__(self, dt=0.002):
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=dt, substeps=10),
            viewer_options=gs.options.ViewerOptions(res=(1280,720)),
        )
        # Add ground
        self.scene.add_entity(gs.morphs.Plane())
        # Add H1 humanoid - will auto-download MJCF
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(file="xml/unitree_h1/h1.xml"), # Genesis includes this
        )
        self.scene.build(n_envs=4096)

    def step(self, actions: torch.Tensor):
        # actions: [4096, 21] -> PD control
        self.robot.control_dofs_position(actions)
        self.scene.step()
        # Return differentiable state
        return {
            "qpos": self.robot.get_qpos(),
            "qvel": self.robot.get_dofs_velocity(),
            "contact_forces": self.scene.rigid_solver.collider.get_contacts(),
            "com": self.robot.get_pos()  # center of mass
        }
