import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE
from scripted_policy import PickAndTransferPolicy, InsertionPolicy

from datetime import datetime


from utils import Log

import IPython
e = IPython.embed


# python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data2 --num_episodes 10 --onscreen_render >> a.txt
# git add . ;git commit -m "Modify record_sim_episodes.py";git push origin main ;


def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_name = args['task_name']
    dataset_dir = args['dataset_dir']
    num_episodes = args['num_episodes']
    onscreen_render = args['onscreen_render']
    inject_noise = False
    render_cam_name_angle = 'angle'
    render_cam_name = 'top'

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    episode_len = SIM_TASK_CONFIGS[task_name]['episode_len']
    camera_names = SIM_TASK_CONFIGS[task_name]['camera_names']
    if task_name == 'sim_transfer_cube_scripted':
        # ATTENTION_HERE
        policy_cls = PickAndTransferPolicy
    elif task_name == 'sim_insertion_scripted':
        policy_cls = InsertionPolicy
    else:
        raise NotImplementedError
    neg_cnt=0
    neg_cnt2=0
    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        # ATTENTION_HERE
        env = make_ee_sim_env(task_name)
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        if onscreen_render:
            # ax = plt.subplot()
            # plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            # plt.ion()
            # print("s")
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt_img = ax1.imshow(ts.observation['images'][render_cam_name])
            plt_img2 = ax2.imshow(ts.observation['images'][render_cam_name_angle])
            plt.ion()

        for step in range(episode_len):
            action = policy(ts) 
            ts = env.step(action) 
            # Log("ee_ts.observation['qpos']\n",ts.observation['qpos'])
            # Log("ee_ts.observation['gripper_ctrl']\n",ts.observation['gripper_ctrl'])
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt_img2.set_data(ts.observation['images'][render_cam_name_angle])
                plt.pause(0.002)

        plt.close()
        Log("len(episode)",len(episode))
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]

        Log("len(joint_traj)",len(joint_traj))
        
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]

        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0




      
        # clear unused variables
        del env
        del episode
        del policy
        Log("#######################################################################################3")
        # setup the environment
        print('Replaying joint commands')
        # ATTENTION_HERE
        env = make_sim_env(task_name)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()


        ##Add by hx
        from PIL import Image
        current_dir = os.path.dirname(os.path.abspath(__file__))  
        cam_folder = os.path.join(current_dir, "cam")  
        if not os.path.exists("cam"):
            os.makedirs(cam_folder)
        def save_cam(image_data,i):
            image = Image.fromarray(image_data)
            filename = os.path.join(cam_folder, f"image_{i}.png")
            image.save(filename)

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][render_cam_name])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            # ATTENTION_HERE
            action = joint_traj[t]
            Log(action)
            ts = env.step(action) 
         
            # Log("ts2.observation['qpos']\n",ts.observation['qpos'])
            # Log("ts2.observation['qvel']\n",ts.observation['qvel'])
            save_cam(ts.observation['images'][render_cam_name],t)
 
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][render_cam_name])
                plt.pause(0.02)
 
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }

        for cam_name in camera_names:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
            
        ### MODIFY
        joint_traj=[jt[7:14] for jt in joint_traj]

        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        # ATTENTION_HERE
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            ### MODIFY
            # data_dict['/observations/qpos'].append(ts.observation['qpos'])
            # data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/observations/qpos'].append(ts.observation['qpos'][7:14])
            data_dict['/observations/qvel'].append(ts.observation['qvel'][7:14])
            data_dict['/action'].append(action)
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # ATTENTION_HERE
        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            ### MODIFY
            # qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            # qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            qpos = obs.create_dataset('qpos', (max_timesteps, 7))
            qvel = obs.create_dataset('qvel', (max_timesteps, 7))

            ### MODIFY
            # action = root.create_dataset('action', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 7))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--task_name', type=str, default='sim_transfer_cube_scripted', help='task_name')

    # parser.add_argument('--dataset_dir', action='store', type=str, help='dataset saving dir', required=True)
    parser.add_argument('--dataset_dir', action='store', type=str, default='data',help='dataset saving dir')
    parser.add_argument('--num_episodes', default=50,type=int, help='num_episodes')
    parser.add_argument('--onscreen_render', action='store_true')
    
    main(vars(parser.parse_args()))

