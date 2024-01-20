import h5py

# Replace 'path/to/your/episode_0.hdf5' with the actual HDF5 file path
dataset_path = '/home/smj/hx/aloha_act/data/sim_transfer_cube_scripted/episode_0.hdf5'

# Open HDF5 file
# with h5py.File(dataset_path, 'r') as file:
#     # Print root attributes
#     print("Root attributes:")
#     print(file.attrs)

    # # Print information about datasets and subgroups in the observations group
    # observations_group = file['observations']
    # print("\nObservations group:")
    # for key in observations_group.keys():
    #     item = observations_group[key]
    #     if isinstance(item, h5py.Group):
    #         # If it's a subgroup, iterate over its keys
    #         print(f"{key}:")
    #         for subkey in item.keys():
    #             subdataset = item[subkey]
    #             print(f"  {subkey}: {subdataset.shape}")
    #     elif isinstance(item, h5py.Dataset):
    #         # If it's a dataset, directly print its shape
    #         print(f"{key}: {item.shape}")

    # # Print information about the action dataset
    # action_dataset = file['action']
    # print("\nAction dataset:")
    # print(f"Shape: {action_dataset.shape}")


with h5py.File("data/sim_transfer_cube_scripted/episode_0.hdf5", 'r') as root:
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        cam_name="top"
        cam=root[f'/observations/images/{cam_name}'][()]
        print((qpos.shape))
        print(cam.shape)
