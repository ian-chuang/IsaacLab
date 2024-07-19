import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--cpu", type=bool, default=True, help="Use CPU for simulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
from env import BlockStackingEnv
import torch

def main():

    num_blocks = 4

    # create the environment
    env = BlockStackingEnv(
        num_blocks=num_blocks,
    )

    env.reset()

    action = torch.tensor([0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0])

    def move_to_pose(simulation_app, env, pose, grasp, max_steps=200):
        for i in range(max_steps):
            action[0:3] = pose[0:3]
            action[7] = 1.0 if grasp else 0.0
            env.step(action)

            if (pose[0:3] - env.get_obs()["ee_pose"][0:3]).norm() < 0.001:
                break

            if not simulation_app.is_running():
                break

    def move_until_touch(simulation_app, env, pose, grasp, max_steps=200, force_threshold=30.0):
        for i in range(max_steps):
            action[0:3] = pose[0:3]
            action[7] = 1.0 if grasp else 0.0
            env.step(action)
            obs = env.get_obs()

            if obs["ee_force"].norm() > force_threshold:
                break
            if (pose[0:3] - obs["ee_pose"][0:3]).norm() < 0.001:
                break
            if not simulation_app.is_running():
                break

    def grasp(env):
        obs = env.get_obs()
        action = torch.zeros(8)
        action[0:7] = obs["ee_pose"]
        action[7] = 1.0
        env.step(action)

    def release(env):
        obs = env.get_obs()
        action = torch.zeros(8)
        action[0:7] = obs["ee_pose"]
        action[7] = 0.0
        env.step(action)

    # run the simulation
    while simulation_app.is_running():
        obs = env.get_obs()

        for i in range(num_blocks):
            block_pose = obs[f"block{i}"]


            # approach
            move_to_pose(simulation_app, env, block_pose + torch.tensor([0.0, 0.0, 0.1]), grasp=False)

            # touch
            move_until_touch(simulation_app, env, block_pose, grasp=False)

            # grasp
            grasp(env)

            # retract
            move_to_pose(simulation_app, env, block_pose + torch.tensor([0.0, 0.0, 0.1]), grasp=True)

            # place approach
            move_to_pose(simulation_app, env, torch.tensor([0.3, 0.2, 0.5]), grasp=True)
            
            # place
            move_until_touch(simulation_app, env, torch.tensor([0.3, 0.2, 0.0]), grasp=True, force_threshold=40)

            # release
            release(env)

            # retract
            move_to_pose(simulation_app, env, torch.tensor([0.3, 0.2, 0.5]), grasp=False)
            

        env.reset()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()