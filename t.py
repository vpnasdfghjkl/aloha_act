import h5py

# Replace 'path/to/your/episode_0.hdf5' with the actual HDF5 file path
dataset_path = '/home/smj/hx/aloha_act/data/sim_transfer_cube_scripted/episode_0.hdf5'

# Open HDF5 file
with h5py.File(dataset_path, 'r') as file:
    # Print root attributes
    print("Root attributes:")
    print(file.attrs)

    # Print information about datasets and subgroups in the observations group
    observations_group = file['observations']
    print("\nObservations group:")
    for key in observations_group.keys():
        item = observations_group[key]
        if isinstance(item, h5py.Group):
            # If it's a subgroup, iterate over its keys
            print(f"{key}:")
            for subkey in item.keys():
                subdataset = item[subkey]
                print(f"  {subkey}: {subdataset.shape}")
        elif isinstance(item, h5py.Dataset):
            # If it's a dataset, directly print its shape
            print(f"{key}: {item.shape}")

    # Print information about the action dataset
    action_dataset = file['action']
    print("\nAction dataset:")
    print(f"Shape: {action_dataset.shape}")


#rec:
#python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data --num_episodes 10 --onscreen_render 
    
#python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir /media/smj/新加卷1/dataset/ckpt_dir_7dim_7_14/ --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 500  --lr 1e-5 --seed 0 


#note:
    # 收集数据的时候取后7维
    # 训练的时候把相关的4个14置为7
    # 测试的时候:
        # imitate_episodes.py line184 取模型平均值的时候填充为初始值:np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239],方差给一个很小很小的值
        # imitate_episodes.py line235 因为训练的时候改了state_dim为7,但是双臂环境需要14,所以需要在设置一次14
        # imitate_episodes.py line271 all_actions = policy(qpos[:, 7:14], curr_image),取张量的后7位,注意第一维度保留.不能是qpos[7:14]
        # imitate_episodes.py line298 raw_action=np.concatenate((np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239]),raw_action )),预测结果是7维,在双臂环境下加上左臂的静态位置
    
