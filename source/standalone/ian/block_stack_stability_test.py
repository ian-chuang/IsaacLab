import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import random
import os

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
import omni.isaac.manipulators

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.utils.math import subtract_frame_transforms

def design_scene(n_cuboids=5):
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    
    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim(f"/World/Origin1", "Xform", translation=origins[0])

    cuboids = []

    sizes = torch.rand(n_cuboids, 3) * 0.05 + 0.1

    for i in range(5):
        cuboid_cfg = RigidObjectCfg(
            prim_path=f"/World/Origin1/Cuboid{i}",
            spawn=sim_utils.CuboidCfg(
                # random size between .1 and .15
                size= sizes[i].tolist(),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                # random mass between 0.5 and 1.5
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5 + 1.0 * torch.rand(1).item()),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # random color
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(random.random(), random.random(), random.random()), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(),
        )
        cuboid_object = RigidObject(cfg=cuboid_cfg)
        cuboids.append(cuboid_object)

    # return the scene information
    scene_entities = {"cuboids": cuboids}
    return scene_entities, origins, sizes


def transform_wrench_to_world_frame(object, wrench):
    """Transforms the force and torque from the object frame to the world frame."""
    # obtain quantities from simulation
    body_pose_w = object.data.body_state_w[0, 0, 0:7]

    body_rot_w = body_pose_w[3:7]

    # convert to rotation matrix
    body_rot_w = math_utils.matrix_from_quat(body_rot_w)

    # transform force from body frame to world frame  (use inverse)
    wrench[:3] = torch.matmul(body_rot_w.t(), wrench[:3])
    wrench[3:] = torch.matmul(body_rot_w.t(), wrench[3:])

    return wrench

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], origins: torch.Tensor, sizes: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cuboids = entities["cuboids"]
    n = len(cuboids)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        if count % 150 == 0:

            buffer = 1e-5

            height = sizes[0,2] * 0.5 + buffer
            for i, cuboid in enumerate(cuboids):
                root_state = cuboid.data.default_root_state.clone()
                root_state[:, :3] += origins
                root_state[:, 2] += height
                root_state[:, 0:2] += torch.randn(2, device=sim.device) * 0.02
                cuboid.write_root_state_to_sim(root_state)
                cuboid.reset()
                height += sizes[i,2] + buffer

            print("----------------------------------------")
            print("[INFO]: Resetting object state...")


        world_wrench = torch.zeros((n, 6), device=sim.device)
        world_wrench[:,2] = -9.8
        wrench = torch.zeros((n, 6), device=sim.device)
        wrench[:, :3] = torch.randn(n, 3) * 4.0

        for i, cuboid in enumerate(cuboids):
            new_world_wrench = transform_wrench_to_world_frame(cuboid, world_wrench[i])
            total_wrench = new_world_wrench + wrench[i]
            cuboid.set_external_force_and_torque(forces=total_wrench[:3], torques=total_wrench[3:])

        for cuboid in cuboids:
            cuboid.write_data_to_sim()

        sim.step()

        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for cuboid in cuboids:
            cuboid.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
        gravity=(0.0, 0.0, 0.0),
    )
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[1.5, 0.0, 1.0], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities, scene_origins, sizes = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins, sizes)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
