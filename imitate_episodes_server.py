import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions

### MODIFY ADD
from utils import Log
from flask import Flask, request, jsonify

### 
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed


# python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpt_dir --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0
# python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir /media/smj/新加卷1/dataset/ckpt_dir_only_right/ --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 100  --lr 1e-5 --seed 0

# python3 imitate_episodes.py 
# --task_name sim_transfer_cube_scripted 
# --ckpt_dir /media/smj/新加卷1/dataset/ckpt_dir_7dim/ 
# --policy_class ACT 
# --kl_weight 10 
# --chunk_size 100 
# --hidden_dim 512 
# --batch_size 8 
# --dim_feedforward 3200 
# --num_epochs 100  
# --lr 1e-5 
# --seed 0 
# --eval 
# --temporal_agg 
# --onscreen_render


def main(args,cam_followed,cam_fixed,current_joints):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    

    camera_names = ['cam_followed','cam_fixed']

    # fixed parameters
    ### MODIFY
    # state_dim = 14
    state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         # ATTENTION_HERE
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,

        'cam_followed':cam_followed,
        'cam_fixed':cam_fixed,
        'current_joints':current_joints
    }
    Log("here")
    if is_eval:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            target_pose = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name,target_pose])
    return target_pose 

   


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image
### MODIFY 
def get_image_from_arm(cam):
    curr_images = []
    for cam_name in cam:
        curr_image = rearrange(cam_name, 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    # real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    # camera_names = config['camera_names']
    # max_timesteps = config['episode_len']
    # task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    # onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    ### MODIFY 
    # stats['qpos_mean'] = np.concatenate((np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239]),stats['qpos_mean']))
    # stats['qpos_std'] = np.concatenate((np.array([1e-2, 1e-2, 1e-2,1e-2,1e-2,1e-2,1e-2]),stats['qpos_std']))
    # stats['action_mean'] = np.concatenate((np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239]),stats['action_mean'] ))
    # stats['action_std'] = np.concatenate((np.zeros(7),stats['action_std'],))
    ### DEL ABOVE

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    ### MODIFY
    # max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks
    max_timesteps = int(1) # may increase for real-world tasks

    num_rollouts = 1


    cam_followed=config['cam_followed']
    cam_fixed=config['cam_fixed']
    cam=[cam_followed,cam_fixed]
    current_joints=config['current_joints']


    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task

        ### onscreen render
        # if onscreen_render:
        #     ax = plt.subplot()
        #     plt_img = ax.imshow(cam_followed)
        #     plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
        ### MODIFY
        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        qpos_history = torch.zeros((1, max_timesteps, 7)).cuda()
        # image_list = [] # for visualization
        # qpos_list = []
        # target_qpos_list = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                # if onscreen_render:
                #     image = cam_followed
                #     plt_img.set_data(image)
                #     # plt.pause(DT)

              
                

                    
                ### MODIFY    
                qpos_numpy = current_joints
                Log("qpos_numpy:",type(qpos_numpy),qpos_numpy.shape)

                ### MODIFY

                Log("cam_followed.shape=",cam_followed.shape,"cam_followed.cam_fixed=",cam_fixed.shape)

                 

                qpos = pre_process(qpos_numpy)

                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                
                qpos_history[:, t] = qpos

                curr_image = get_image_from_arm(cam)
                Log("curr_image:",curr_image.shape)

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        ### MODIFY 
                        print(qpos.shape)
                        # all_actions = policy(qpos, curr_image)
                        all_actions = policy(qpos, curr_image)
                        print("all_actions.shape",all_actions.shape)
                        
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action
                print(target_qpos)
  

                ### for visualization
                # qpos_list.append(qpos_numpy)
                # target_qpos_list.append(target_qpos)

            # plt.close()

        # save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
    
    return target_qpos


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


app = Flask(__name__)
@app.route('/process_data', methods=['POST'])
def process_data():
    print("server start#####################################################3")
    data_dict = request.get_json()
    cam_followed = np.array(data_dict['cam_followed'],dtype=np.uint8)
    cam_fixed = np.array(data_dict['cam_fixed'],dtype=np.uint8)
    current_joints = np.array(data_dict['pose'])
    Log(cam_followed.shape,'\n',cam_fixed.shape,'\n',current_joints.shape)
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True,default="/media/smj/新加卷1/dataset/ckpt_dir_7dim_7_14/")
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True,default="ACT")
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True, default=32)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True,default=0)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True, default=10)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True,default=1e-5)

    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200)
    parser.add_argument('--temporal_agg', action='store_true')


    result=main(vars(parser.parse_args()),cam_followed,cam_fixed,current_joints)

    return jsonify(result.tolist())  

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777)