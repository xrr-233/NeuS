import os
os.environ['PATH'] = "/home/bsft19/ruixu33/Programs/anaconda3/bin:/home/bsft19/ruixu33/Programs/anaconda3/condabin:/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/games:/usr/local/games:/home/bsft19/ruixu33/Programs/anaconda3/bin"

import logging
import argparse
import torch
import matplotlib.pyplot as plt
from models.runner import Runner

def visualize_intrinsic():
    rays_o, rays_v = runner.dataset.gen_rays_at(0, resolution_level=4)
    rays_o = rays_o.cpu().numpy()
    rays_v = rays_v.cpu().numpy()
    print(rays_o[0, 0])
    print(rays_v[0, 0])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    x = rays_v[:, :, 0]
    y = rays_v[:, :, 1]
    z = rays_v[:, :, 2]
    ax.scatter(x, y, z)
    x = rays_o[0, 0, 0]
    y = rays_o[0, 0, 1]
    z = rays_o[0, 0, 2]
    ax.scatter(x, y, z)
    plt.show()

if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/womask.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=True, action="store_true") # default=false
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='haibao/preprocessed')

    parser.add_argument('--train_resolution', type=int, default=64)
    parser.add_argument('--validate_resolution', type=int, default=512)  # Higher value, clearer effect
    # For rendering
    parser.add_argument('--render_resolution', type=float, default=4)  # Lower value, clearer effect
    parser.add_argument('--render_step', type=int, default=60)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)
    # visualize_intrinsic()

    # runner.funky_town(resolution_level=args.render_resolution, n_frames=args.render_step)
    # runner.validate_mesh(world_space=True, resolution=args.validate_resolution, threshold=args.mcube_threshold)
    runner.interpolate_view(0, 26, resolution_level=args.render_resolution, n_frames=args.render_step)

    '''if args.mode == 'train':
        runner.train(resolution=args.train_resolution)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=args.validate_resolution, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1, resolution_level=args.render_resolution, n_frames=args.render_step)'''
