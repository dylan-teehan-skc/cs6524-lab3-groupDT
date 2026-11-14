import cv2 as cv
import numpy as np
import yaml

from mypylibs.dense import dense_cloud, write_ply
from mypylibs.sfm import keyframes, load_frames, sfm


def scale_focal(K, factor):
    K = K.copy()
    K[[0, 1], [0, 1]] *= factor
    return K


def run_one(name, frames, K, D, mode, min_move):
    print(name)
    imgs = [cv.undistort(f, K, D) for f in frames] if mode == 'u' else frames
    poses, sparse, err = sfm(imgs, K, min_move)
    xyz, rgb = dense_cloud(imgs, K, poses, sparse, still_camera=min_move > 0)
    write_ply(f'out/{name}.ply', xyz, rgb)
    print(f'got a camera for {len(poses)} of {len(frames)} frames, error {err:.2f}px')
    print(f'{len(sparse)} points in the sparse cloud, {len(xyz)} in the dense one')


def run(only=None):
    K = np.loadtxt('in/K.csv', delimiter=',')
    D = np.loadtxt('in/D.csv', delimiter=',')
    for video in 'ab':
        with open(f'in/{video}.yaml') as f:
            params = yaml.safe_load(f)
        frames = keyframes(load_frames(f'in/{video}.mp4', params['step']), params['min_move'])
        for factor in (0.8, 1.0, 1.2):
            for mode in 'ud':
                name = f'{video}_{round(factor * 10):02}{mode}'
                if not only or name == only:
                    run_one(name, frames, scale_focal(K, factor), D, mode, params['min_move'])
