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

    # run the simulation
    while simulation_app.is_running():
        obs = env.get_obs()

        for i in range(num_blocks):
            block_pose = obs[f"block{i}"]


            # approach
            for i in range(150):
                action[0:3] = block_pose[0:3] + torch.tensor([0.0, 0.0, 0.1])
                action[7] = 0.0
                env.step(action)

            # touch
            for i in range(150):
                action[0:3] = block_pose[0:3] 
                action[7] = 0.0
                env.step(action)
                obs = env.get_obs()
                if obs["ee_force"].norm() > 20.0:
                    break

            # grasp
            action[0:3] = block_pose[0:3] 
            action[7] = 1.0
            env.step(action)

            # retract
            for i in range(150):
                action[0:3] = block_pose[0:3] + torch.tensor([0.0, 0.0, 0.1])
                action[7] = 1.0
                env.step(action)

            # rest
            for i in range(150):
                action[0:3] = torch.tensor([0.5, 0.0, 0.5])
                action[7] = 1.0
                env.step(action)

            # place approach
            for i in range(150):
                action[0:3] = torch.tensor([0.3, 0.0, 0.4])
                action[7] = 1.0
                env.step(action)
                print( obs["ee_force"].norm())
            
            # place
            for i in range(150):
                action[0:3] = torch.tensor([0.3, 0.0, 0.0])
                action[7] = 1.0
                env.step(action)
                obs = env.get_obs()
                print( obs["ee_force"].norm())
                if obs["ee_force"].norm() > 20.0:
                    break

            # release
            action[0:3] = torch.tensor([0.3, 0.0, 0.0])
            action[7] = 0.0
            env.step(action)

            # retract
            for i in range(150):
                action[0:3] = torch.tensor([0.3, 0.0, 0.4])
                action[7] = 0.0
                env.step(action)
            

        env.reset()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()