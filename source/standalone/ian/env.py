import torch
import random
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, AssetBase, RigidObject, Articulation
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from ur5e_picke import UR5e_PICKe_CFG
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper

@configclass
class BlockStackingSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, num_envs, env_spacing, num_blocks):
        super().__init__(num_envs=num_envs, env_spacing=env_spacing)
        # ground plane
        self.ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        )

        # lights
        self.dome_light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        )

        # Table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
            spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
        )

        # articulation
        self.robot = UR5e_PICKe_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # blocks
        for i in range(num_blocks):
            # set block attr
            setattr(self, f"block{i}", RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Block" + str(i),
                spawn=sim_utils.CuboidCfg(
                    # random size between .1 and .15
                    size= (0.05, 0.05, 0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    # random mass between 0.5 and 1.5
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5 + 1.0 * torch.rand(1).item()),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    # random color
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(random.random(), random.random(), random.random()), metallic=0.2)
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.2 + 0.1 * i, -0.2, 0.025)),
            ))

        # self.camera = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/zedm_left_camera_frame/zedm_left_camera_optical_frame",
        #     update_period=0.03,
        #     height=480,
        #     width=640,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        #     ),
        #     offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.017), rot=(0,1,0,0), convention="ros"),
        # )

        self.ee_contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/suction", update_period=0.0, history_length=5, debug_vis=True
        )

def limit_distance(cur_pos, target_pos, max_dist):
    dist = torch.norm(target_pos - cur_pos)
    if dist > max_dist:
        return cur_pos + (target_pos - cur_pos) * max_dist / dist
    return target_pos


class BlockStackingEnv:

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
 
        # Load kit helper
        self.sim_cfg = sim_utils.SimulationCfg(
            use_gpu_pipeline=False,
            device="cpu",
        )
        self.sim = SimulationContext(self.sim_cfg)
        # Set main camera
        self.sim.set_camera_view(eye=(1.5, 0.0, 1.0), target=(0.0, 0.0, 0.0))

        # Design scene
        self.scene_cfg = BlockStackingSceneCfg(
            num_envs=1, 
            env_spacing=0.0,
            num_blocks=num_blocks
        )

        self.scene = InteractiveScene(self.scene_cfg)

        self.sim.reset()
        self.sim_dt = self.sim.get_physics_dt()

        # done building sim
        self.robot = self.scene["robot"]

        self.blocks = [self.scene[f"block{i}"] for i in range(num_blocks)]

        self.robot_entity_cfg = SceneEntityCfg(
            "robot", 
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint", 
                "wrist_2_joint", 
                "wrist_3_joint",
            ],
            body_names=["tcp"]
        )
        # Resolving the scene entities
        self.robot_entity_cfg.resolve(self.scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Create controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)

        self.gripper = SurfaceGripper(
            end_effector_prim_path="/World/envs/env_0/Robot/tcp",
            translate=0.01,
            direction="z",
        )
        self.gripper.initialize()

        # Markers
        self.frame_marker_cfg = FRAME_MARKER_CFG.copy()
        self.frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(self.frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(self.frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))


        # reset actions
        self.ee_goal = torch.tensor([0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0], device=self.sim.device)

    def step(self, action: torch.Tensor):
        # obtain quantities from simulation
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        root_pose_w = self.robot.data.root_state_w[:, 0:7]
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute frame in root frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        #limit action translation to 0.1 by finding diff from ee_pos_b
        action[0:3] = limit_distance(ee_pos_b.reshape(3), action[0:3], 0.05)

        # DIFF IK
        self.diff_ik_controller.set_command(action[0:7].reshape(1, -1))

        # compute the joint commands
        joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # GRIPPER
        gripper_cmd = action[7]
        if gripper_cmd > 0.5 and not self.gripper.is_closed():
            self.gripper.close()
        elif gripper_cmd < 0.5 and self.gripper.is_closed():
            self.gripper.open()


        # SIM STUFF

        # apply actions
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()
        # perform step
        self.sim.step()
        # update buffers
        self.scene.update(self.sim_dt)

        # MARKERS

        # obtain quantities from simulation
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(action[0:3].reshape(1, -1) + self.scene.env_origins, action[3:7].reshape(1, -1))

    def reset(self):
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        for block in self.blocks:
            root_state = block.data.default_root_state.clone()
            root_state[:, :3] += self.scene.env_origins
            block.write_root_state_to_sim(root_state)
            block.reset()

        # reset controller
        self.diff_ik_controller.reset()

        # reset gripper
        if self.gripper.is_closed():
            self.gripper.open()
            

    def get_obs(self):
        obs = {}

        for i, block in enumerate(self.blocks):
            object_data = block.data
            object_position = object_data.root_pos_w - self.scene.env_origins
            obs[f"block{i}"] = object_position[0]

        obs["ee_force"] = self.scene["ee_contact_forces"].data.net_forces_w.reshape(-1)

        obs["ee_pose"] = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7].reshape(-1)

        return obs












# Table
# self.table_cfg = UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
# self.table_cfg.func("/World/Table", self.table_cfg, translation=(0.5, 0.0, 0.0), orientation=(0.707, 0.0, 0.0, 0.707))

# # plane
# self.plane_cfg = GroundPlaneCfg()
# self.plane_cfg.func("/World/GroundPlane", self.plane_cfg, translation=(0.0, 0.0, -1.05))

# # spawn distant light
# self.cfg_light_distant = sim_utils.DistantLightCfg(
#     intensity=3000.0,
#     color=(0.75, 0.75, 0.75),
# )
# self.cfg_light_distant.func("/World/lightDistant", self.cfg_light_distant, translation=(1, 0, 5))

# # robot
# self.robot_cfg = UR5e_PICKe_CFG.copy()
# self.robot_cfg.prim_path = "/World/Robot"
# self.robot = Articulation(cfg=self.robot_cfg)

# # blocks
# self.block_cfgs = []
# self.blocks = []
# for i in range(num_blocks):
#     block_cfg = RigidObjectCfg(
#         prim_path=f"/World/Origin1/Cuboid{i}",
#         spawn=sim_utils.CuboidCfg(
#             # random size between .1 and .15
#             size=(0.1, 0.1, 0.1),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(),
#             # random mass between 0.5 and 1.5
#             mass_props=sim_utils.MassPropertiesCfg(mass=0.5 + 1.0 * torch.rand(1).item()),
#             collision_props=sim_utils.CollisionPropertiesCfg(),
#             # random color
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0,0), metallic=0.2),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(),
#     )
#     block = RigidObject(cfg=block_cfg)

#     self.block_cfgs.append(block_cfg)
#     self.blocks.append(block)