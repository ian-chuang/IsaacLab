# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app














import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

UR5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"ur5_hybrid_sys_instanceable_meshes.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0,
            "shoulder_lift_joint": -1.57,
            "elbow_joint": 1.57,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "finger_joint": 0.0,
            "right_outer_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0,
            "right_inner_finger_joint": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr= [
                'shoulder_pan_joint',
                'shoulder_lift_joint',
                'elbow_joint',
                'wrist_1_joint',
                'wrist_2_joint',
                'wrist_3_joint'
            ],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "right_outer_knuckle_joint",
            ],
            velocity_limit=10.0,
            effort_limit=4,
            stiffness=4,
            damping=0.5,
        ),
        "passive_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            velocity_limit=10.0,
            effort_limit=0.5,
            stiffness=1,
            damping=0.2,
        ),

    },
)



















"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms

@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
    )

    block = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Block",
        spawn=sim_utils.CuboidCfg(
            size=[0.04, 0.04, 0.04],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=2000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.02]),
    )




    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    block = scene["block"]

    # Create controller
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))



    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)


    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["tcp"])

    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    # Obtain the frame index of the end-effector
    # For a fixed base robot, the frame index is one less than the body index. This is because
    # the root body is not included in the returned Jacobians.
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]





    gripper_entity_cfg = SceneEntityCfg(
        "robot", 
        joint_names=[
            "finger_joint",
            "right_outer_knuckle_joint",
        ])
    gripper_entity_cfg.resolve(scene)

    passive_gripper_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "left_inner_finger_joint",
            "right_inner_finger_joint",
        ])
    passive_gripper_entity_cfg.resolve(scene)



    

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 400 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

            gripper_pos_des = joint_pos[:, gripper_entity_cfg.joint_ids].clone()
            passive_gripper_pos_des = joint_pos[:, passive_gripper_entity_cfg.joint_ids].clone()

            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

            # reset block
            root_state = block.data.default_root_state.clone()
            root_state[:, 0:3] += scene.env_origins
            block.write_root_state_to_sim(root_state)
            block.reset()

        else:


            if count < 50:    
                ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 0 * torch.ones_like(gripper_pos_des)
            elif count < 100:
                ik_commands[:] = torch.tensor([0.5, 0, 0.02, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 0 * torch.ones_like(gripper_pos_des)
            elif count < 150:
                ik_commands[:] = torch.tensor([0.5, 0, 0.02, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 45 * 3.14/180 * torch.ones_like(gripper_pos_des)
            elif count < 200:
                ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 45 * 3.14/180 * torch.ones_like(gripper_pos_des)
            elif count < 225:
                ik_commands[:] = torch.tensor([0.6, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 45 * 3.14/180 * torch.ones_like(gripper_pos_des)
            elif count < 250:
                ik_commands[:] = torch.tensor([0.4, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 45 * 3.14/180 * torch.ones_like(gripper_pos_des)
            elif count < 275:
                ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 45 * 3.14/180 * torch.ones_like(gripper_pos_des)
            elif count < 300:
                ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 0* torch.ones_like(gripper_pos_des)
            elif count < 350:
                ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 45 * 3.14/180 * torch.ones_like(gripper_pos_des)
            elif count < 400:
                ik_commands[:] = torch.tensor([0.5, 0, 0.1, 0, -0.7071068, 0, 0.7071068], device=robot.device).unsqueeze(0)
                gripper_pos_des = 0* torch.ones_like(gripper_pos_des)
            

            diff_ik_controller.set_command(ik_commands)

            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_link_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            passive_gripper_pos_des = -45 * torch.ones_like(passive_gripper_pos_des)
            

            

        # apply actions
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)


        robot.set_joint_position_target(gripper_pos_des, joint_ids=gripper_entity_cfg.joint_ids)
        robot.set_joint_position_target(passive_gripper_pos_des, joint_ids=passive_gripper_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # obtain quantities from simulation
        ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()