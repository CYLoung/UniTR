import yaml
import pickle
from pathlib import Path
from easydict import EasyDict
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
import numpy as np
from tqdm import tqdm

def get_available_scenes(nusc):
    available_scenes = []
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        num_lidar_pts = 0

        sample_token = scene_rec['first_sample_token']
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            # Read the point cloud file to count points
            lidar_path = nusc.get_sample_data_path(sd_rec['token'])
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            num_lidar_pts += points.shape[0]
            sample_token = sample['next']

        if num_lidar_pts > 0:
            available_scenes.append(scene)

    print(f"Found {len(available_scenes)} available scenes.")
    return available_scenes
def fill_trainval_infos(nusc, train_scenes, val_scenes, data_path, max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []

    for scene in tqdm(nusc.scene):
        scene_token = scene['token']
        if scene_token not in train_scenes and scene_token not in val_scenes:
            continue

        sample_token = scene['first_sample_token']
        while sample_token != '':
            sample = nusc.get('sample', sample_token)
            sample['scene_token'] = scene_token  # Add scene token to sample
            lidar_sd_token = sample['data']['LIDAR_TOP']
            lidar_sd_rec = nusc.get('sample_data', lidar_sd_token)
            sample['lidar_path'] = nusc.get_sample_data_path(lidar_sd_rec['token'])
            
            # Add sweeps information
            sample['sweeps'] = []
            for _ in range(max_sweeps):
                lidar_sweep_token = lidar_sd_rec['prev']
                if lidar_sweep_token == '':
                    break
                lidar_sweep_rec = nusc.get('sample_data', lidar_sweep_token)
                ego_pose = nusc.get('ego_pose', lidar_sweep_rec['ego_pose_token'])
                sweep_info = {
                    'lidar_path': nusc.get_sample_data_path(lidar_sweep_rec['token']),
                    'transform_matrix': ego_pose['translation'],
                    'time_lag': sample['timestamp'] - lidar_sweep_rec['timestamp']
                }
                sample['sweeps'].append(sweep_info)
                lidar_sd_rec = lidar_sweep_rec

            if scene_token in train_scenes:
                train_nusc_infos.append(sample)
            elif scene_token in val_scenes:
                val_nusc_infos.append(sample)
            sample_token = sample['next']

    return train_nusc_infos, val_nusc_infos



def create_nuscenes_info(version, data_path, save_path, max_sweeps=10):
    data_path = Path(data_path) / version
    save_path = Path(save_path) / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print(f'{version}: train scene({len(train_scenes)}), val scene({len(val_scenes)})')

    train_nusc_infos, val_nusc_infos = fill_trainval_infos(
        nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        data_path=data_path, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print(f'test sample: {len(train_nusc_infos)}')
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print(f'train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}')
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)

def main():
    with open('cfgs/dataset_configs/nuscenes_dataset.yaml', 'r') as f:
        dataset_cfg = EasyDict(yaml.safe_load(f))

    ROOT_DIR = Path('/root/src/nuScenes')
    dataset_cfg.VERSION = 'v1.0-mini'
    dataset_cfg.DATA_PATH = str(ROOT_DIR / 'v1.0-mini')
    dataset_cfg.INFO_PATH = {
        'train': ['nuscenes_infos_10sweeps_train.pkl'],
        'test': ['nuscenes_infos_10sweeps_val.pkl'],
    }

    create_nuscenes_info(
        version=dataset_cfg.VERSION,
        data_path=ROOT_DIR,
        save_path=ROOT_DIR,
        max_sweeps=dataset_cfg.MAX_SWEEPS
    )

if __name__ == '__main__':
    main()

