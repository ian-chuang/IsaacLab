import torch
from omni.isaac.lab.utils.math import (
    quat_from_angle_axis, 
    matrix_from_quat, 
    quat_from_matrix, 
    axis_angle_from_quat,
    apply_delta_pose
)

def transform_from_pos_quat(pos, quat):
    rot = matrix_from_quat(quat)

    t = torch.eye(4, device=pos.device).repeat(pos.shape[0], 1, 1)
    t[:, :3, :3] = rot
    t[:, :3, 3] = pos

    return t


def transform_from_spatial_vector(spatial_vector):
    pos = spatial_vector[:, :3]
    rotvec = spatial_vector[:, 3:]

    angle = torch.norm(rotvec, dim=1)

    # Avoid division by zero
    angle = angle + 1e-6

    axis = rotvec / angle.unsqueeze(1)
    quat = quat_from_angle_axis(angle, axis)
    rot = matrix_from_quat(quat)

    return transform_from_pos_quat(pos, quat)

def spatial_vector_from_transform(transform):
    rot = transform[:, :3, :3]
    pos = transform[:, :3, 3]

    quat = quat_from_matrix(rot)
    rotvec = axis_angle_from_quat(quat)

    return torch.cat([pos, rotvec], dim=1)

def apply_dead_band_smooth(component, bandwidth, smooth_band):
    """
    component: batch of vector of size (N, M)
    bandwidth: float
    smooth_band: float
    """
    abs_component = torch.abs(component)
    mask_dead = abs_component <= bandwidth
    mask_smooth = (abs_component > bandwidth) & (abs_component <= (bandwidth + smooth_band))
    mask_pass = abs_component > (bandwidth + smooth_band)

    smooth_result = torch.zeros_like(component, device=component.device)

    smooth_result[mask_dead] = 0

    smooth_result[mask_pass] = (abs_component[mask_pass] - bandwidth - smooth_band * 0.5) / abs_component[mask_pass]

    s = abs_component[mask_smooth] - bandwidth
    smooth_result[mask_smooth] = (0.5 * s * s / smooth_band) / abs_component[mask_smooth]

    return smooth_result * component

def wrench_dead_band_smooth(wrench_in, bandwidth_force, bandwidth_torque, smooth_force, smooth_torque):
    """
    Apply the smooth dead band to each component of the wrench input
    (Torch version)
    wrench_in: 6D vector of size (N, 6)
    bandwidth_force: float
    bandwidth_torque: float
    smooth_force: float
    smooth_torque: float
    """
    assert wrench_in.shape[1] == 6, "wrench_in must be a 6D vector"

    # Separate force and torque components
    force = wrench_in[:, :3]
    torque = wrench_in[:, 3:]

    # Apply smoothing to force and torque components
    smooth_force_result = apply_dead_band_smooth(force, bandwidth_force, smooth_force)
    smooth_torque_result = apply_dead_band_smooth(torque, bandwidth_torque, smooth_torque)

    # Combine the results
    wrench_out = torch.cat((smooth_force_result, smooth_torque_result), dim=1)

    return wrench_out

def wrench_trans(T_from_to, w_from):
    """
    Transforms a wrench to a new point of view.

    Args:
    T_from_to: The transformation to the new point of view (Pose) 
                represented as a 4x4 homogeneous transformation matrix. (N, 4, 4)
    w_from: Wrench to transform in list format (N, 6) [F_x, F_y, F_z, M_x, M_y, M_z]

    Returns:
    resulting wrench, w_to in list format (N, 6) [F_x, F_y, F_z, M_x, M_y, M_z]
    """

    T_inv = torch.inverse(T_from_to)
    t0 = T_inv[:, :3, 3]
    r0 = T_inv[:, :3, :3]

    F = w_from[:, :3]
    M = w_from[:, 3:]

    F_to = torch.matmul(r0, F.unsqueeze(2)).squeeze(2)
    M_to = torch.matmul(r0, M.unsqueeze(2)).squeeze(2) + torch.cross(t0, F_to, dim=1)
    w_to = torch.cat((F_to, M_to), dim=1)
    return w_to

def adm_rotate_velocity_in_frame(frame, velocity):
    """
    Rotates a velocity into a new reference frame.

    :param frame: List or array of the current velocity reference frame (N, 4, 4)
    :param velocity: List or array of the input velocity vector (N, 6) [vx, vy, vz, vrx, vry, vrz]
    :returns: List of the velocity in the new reference frame (N, 6) [vx, vy, vz, vrx, vry, vrz]
    """
    r0 = frame[:, :3, :3]
    vel_pos = velocity[:, :3]
    vel_rot = velocity[:, 3:]

    vel_pos_new = torch.matmul(r0, vel_pos.unsqueeze(2)).squeeze(2)
    vel_rot_new = torch.matmul(r0, vel_rot.unsqueeze(2)).squeeze(2)

    return torch.cat((vel_pos_new, vel_rot_new), dim=1)


def adm_rotate_wrench_in_frame(frame, wrench):
    return adm_rotate_velocity_in_frame(frame, wrench)

def pose_error(desired, current):
    """
    Calculate the pose error between two poses.
    desired: The desired pose (4x4 matrix) (N, 4, 4)
    current: The current pose (4x4 matrix) (N, 4, 4)
    """
    rc1 = current[:, 0:3, 0]
    rc2 = current[:, 0:3, 1]
    rc3 = current[:, 0:3, 2]
    rd1 = desired[:, 0:3, 0]
    rd2 = desired[:, 0:3, 1]
    rd3 = desired[:, 0:3, 2]

    error = torch.zeros((desired.shape[0], 6), device=desired.device)
    error[:, 0:3] = desired[:, 0:3, 3] - current[:, 0:3, 3]
    error[:, 3:6] = 0.5 * (torch.cross(rc1, rd1, dim=1) + torch.cross(rc2, rd2, dim=1) + torch.cross(rc3, rd3, dim=1))

    return error

def adm_vel_trans(t, v):
    v_swap = torch.zeros_like(v, device=v.device)
    v_swap[:, :3] = v[:, 3:]
    v_swap[:, 3:] = v[:, :3]

    vw = wrench_trans(t, v_swap)

    vw_swap = torch.zeros_like(vw, device=vw.device)
    vw_swap[:, :3] = vw[:, 3:]
    vw_swap[:, 3:] = vw[:, :3]

    return vw_swap

def apply_delta_transform(transform, delta):
    delta_transform = transform_from_spatial_vector(delta)
    return torch.matmul(transform, delta_transform)


class ComplianceControllerCfg:
    # general params
    step_time=1.0/125
    flange_to_tcp_frame = [0, 0, 0.3, 0, 0, 0]
    debug = False

    # compliance control params
    mass_scaling=0.5
    damping_scaling=0.5
    mass_list=[22.5, 22.5, 22.5, 1, 1, 1]
    damping_list=[25, 25, 25, 2, 2, 2]
    base_to_compliance_frame=[0, 0, 0, 0, 0, 0]
    tool_flange_to_compliance_center=[0, 0, 0, 0, 0, 0]
    dead_band=[2, 0.15, 2, 0.15]
    compliance_vector=[1, 1, 1, 1, 1, 1]
    stiffness_params=[500, 500, 500, 1, 1, 1]
    max_spring_wrench=[200, 200, 200, 1, 1, 1]

    def validate(self, device):
        assert self.step_time > 0, "step_time must be positive"
        assert len(self.flange_to_tcp_frame) == 6, "flange_to_tcp_frame must be a 6D vector"
        assert self.mass_scaling > 0, "mass_scaling must be positive"
        assert self.damping_scaling > 0, "damping_scaling must be positive"
        assert len(self.mass_list) == 6, "mass_list must be a 6D vector"
        assert len(self.damping_list) == 6, "damping_list must be a 6D vector"
        assert len(self.base_to_compliance_frame) == 6, "base_to_compliance_frame must be a 6D vector"
        assert len(self.tool_flange_to_compliance_center) == 6, "tool_flange_to_compliance_center must be a 6D vector"
        assert len(self.dead_band) == 4, "dead_band must be a 4D vector"
        assert len(self.compliance_vector) == 6, "compliance_vector must be a 6D vector"
        assert len(self.stiffness_params) == 6, "stiffness_params must be a 6D vector"
        assert len(self.max_spring_wrench) == 6, "max_spring_wrench must be a 6D vector"

        def to_torch(x):
            return torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        
        self.flange_to_tcp_frame = transform_from_spatial_vector(to_torch(self.flange_to_tcp_frame))
        self.mass_list = to_torch(self.mass_list)
        self.damping_list = to_torch(self.damping_list)
        self.base_to_compliance_frame = transform_from_spatial_vector(to_torch(self.base_to_compliance_frame))
        self.tool_flange_to_compliance_center = transform_from_spatial_vector(to_torch(self.tool_flange_to_compliance_center))
        self.compliance_vector = to_torch(self.compliance_vector)
        self.stiffness_params = to_torch(self.stiffness_params)
        self.max_spring_wrench = to_torch(self.max_spring_wrench)


class ComplianceController:
    def __init__(self, config, num_envs: int, device: torch.device):
        config.validate(device)
        self.config = config
        self.num_envs = num_envs
        self.device = device

    def reset(self):
        self.x_e = torch.zeros(self.num_envs, 6, device=self.device)
        self.dx_e = torch.zeros(self.num_envs, 6, device=self.device)
        self.ddx_e = torch.zeros(self.num_envs, 6, device=self.device)
        self.last_ddx_e = self.ddx_e.clone()
        self.last_dx_e = self.dx_e.clone()
        self.last_x_e = self.x_e.clone()

    def set_command(self, compliance_to_target_tcp_frame, target_wrench_at_compliance):
        """
        cmd_pose (N, 4, 4)
        cmd_wrench (N, 6)
        """
        self.compliance_to_target_tcp_frame = compliance_to_target_tcp_frame
        self.target_wrench_at_compliance = target_wrench_at_compliance

    def compute(self, base_to_tcp_frame, wrench_at_flange):
        """
        ee_pose (N, 4, 4)
        flange_wrench (N, 6)
        """

        # Apply dead band and smooth
        wrench_at_tool = wrench_dead_band_smooth(
            wrench_at_flange,
            self.config.dead_band[0], self.config.dead_band[1], self.config.dead_band[2], self.config.dead_band[3]
        )

        # Compute the wrench at the compliance center
        wrench_at_compliance_center = wrench_trans(self.config.tool_flange_to_compliance_center, wrench_at_tool)

        # compute transforms
        T_compliance_center_to_tcp = torch.matmul(torch.inverse(self.config.tool_flange_to_compliance_center), self.config.flange_to_tcp_frame)
        T_compliance_center_to_base = torch.matmul(T_compliance_center_to_tcp, torch.inverse(base_to_tcp_frame))
        T_compliance_frame_to_compliance_center = torch.inverse(torch.matmul(T_compliance_center_to_base, self.config.base_to_compliance_frame))

        # calc force and torque error in compliance frame
        wrench_at_compliance = adm_rotate_wrench_in_frame(T_compliance_frame_to_compliance_center, wrench_at_compliance_center)

        # cancel out specific force and torque directions
        wrench_at_compliance = wrench_at_compliance * self.config.compliance_vector

        compliance_to_tcp_frame = torch.matmul(torch.inverse(self.config.base_to_compliance_frame), base_to_tcp_frame)

        # calc spring wrench for compliance
        pose_err = pose_error(compliance_to_tcp_frame, self.compliance_to_target_tcp_frame)
        spring_wrench = pose_err * self.config.stiffness_params
        spring_wrench = torch.clip(spring_wrench, -self.config.max_spring_wrench, self.config.max_spring_wrench)

        self.ddx_e = 1/(self.config.mass_list*self.config.mass_scaling) * \
            (
                (wrench_at_compliance - self.target_wrench_at_compliance) - spring_wrench - 
                (self.dx_e * self.config.damping_list * self.config.damping_scaling)
            )
        self.dx_e = (self.config.step_time * 0.5) * (self.ddx_e + self.last_ddx_e) + self.last_dx_e
        self.x_e = (self.config.step_time * 0.5) * (self.dx_e + self.last_dx_e) + self.last_x_e

        self.last_ddx_e = self.ddx_e.clone()
        self.last_dx_e = self.dx_e.clone()
        self.last_x_e = self.x_e.clone()

        vel_target_flange = adm_rotate_velocity_in_frame(torch.inverse(T_compliance_frame_to_compliance_center), self.dx_e)
        vel_target_tcp = adm_vel_trans(torch.inverse(T_compliance_center_to_tcp), vel_target_flange)
        vel_target_base_tcp = adm_rotate_velocity_in_frame(base_to_tcp_frame, vel_target_tcp)

        return vel_target_base_tcp








       