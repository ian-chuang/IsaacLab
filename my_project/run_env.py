














# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
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
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
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
            velocity_limit=10000.0,
            effort_limit=300000.0,
            stiffness=0.0,
            damping=10000.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "right_outer_knuckle_joint",
            ],
            velocity_limit=10.0,
            effort_limit=2,
            stiffness=4,
            damping=0.5,
        ),
        "passive_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_inner_finger_joint",
                "right_inner_finger_joint",
            ],
            velocity_limit=10.0,
            effort_limit=0.8,
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
from controller import ComplianceController, ComplianceControllerCfg, transform_from_pos_quat
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import ContactSensorCfg


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

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", update_period=0.0, history_length=6, debug_vis=True
    )




    robot = UR5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    block = scene["block"]

    # Create controller
    # diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    # diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)
    compliance_cfg = ComplianceControllerCfg()
    compliance_cfg.step_time = sim.get_physics_dt()
    compliance_cfg.damping_scaling = 7
    compliance_cfg.stiffness_params = [1000, 1000, 1000, 1000, 1000, 1000]
    compliance_cfg.max_spring_wrench = [100, 100, 100, 100, 100, 100]
    compliance_controller = ComplianceController(compliance_cfg, num_envs=scene.num_envs, device=sim.device)


    # Markers
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

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



    target_pos = torch.tensor([0.5, 0, 0.1], device=robot.device).repeat(scene.num_envs, 1)
    target_quat = torch.tensor([0, -0.7071068, 0.7071068, 0], device=robot.device).repeat(scene.num_envs, 1)
    compliance_to_target_tcp_frame = transform_from_pos_quat(target_pos, target_quat)
    target_wrench_at_compliance = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=robot.device).repeat(scene.num_envs, 1)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 300 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            joint_vel_des = torch.zeros_like(joint_pos_des)

            gripper_pos_des = joint_pos[:, gripper_entity_cfg.joint_ids].clone()
            passive_gripper_pos_des = joint_pos[:, passive_gripper_entity_cfg.joint_ids].clone()

            # reset controller
            compliance_controller.reset()
            compliance_controller.set_command(compliance_to_target_tcp_frame, target_wrench_at_compliance)

            # reset block
            root_state = block.data.default_root_state.clone()
            root_state[:, 0:3] += scene.env_origins
            block.write_root_state_to_sim(root_state)
            block.reset()

        else:
            passive_gripper_pos_des = -45 * torch.ones_like(passive_gripper_pos_des)

            if count < 150:    
                gripper_pos_des = 0 * torch.ones_like(gripper_pos_des)
            else:
                gripper_pos_des = 45 * torch.ones_like(gripper_pos_des)
            

            compliance_controller.set_command(compliance_to_target_tcp_frame, target_wrench_at_compliance)

            # obtain quantities from simulation
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_link_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_link_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the velocity target with controller
            base_to_tcp_frame = transform_from_pos_quat(ee_pos_b, ee_quat_b)
            wrench_at_flange = torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.float32, device=robot.device).repeat(scene.num_envs, 1)
            vel_target_base_tcp = compliance_controller.compute(base_to_tcp_frame, wrench_at_flange)

            # dls
            lambda_val = 0.01
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=robot.device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ vel_target_base_tcp.unsqueeze(-1)
            )
            joint_vel_des = delta_joint_pos.squeeze(-1)

        # apply actions
        # robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_velocity_target(joint_vel_des, joint_ids=robot_entity_cfg.joint_ids)

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
        goal_marker.visualize(target_pos + scene.env_origins, target_quat)

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