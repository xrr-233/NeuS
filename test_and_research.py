import os
import logging
import argparse
import trimesh

import numpy as np
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

def set_vertex_color(runner, resolution, threshold):
    bound_min = torch.tensor(runner.dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(runner.dataset.object_bbox_max, dtype=torch.float32)

    vertices, triangles = runner.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
    vertices = np.array(vertices * runner.dataset.scale_mats_np[0][0, 0] + runner.dataset.scale_mats_np[0][:3, 3][None], dtype=np.float32)

    appearance_feature = runner.sdf_network.sdf_hidden_appearance(torch.tensor(vertices))
    surface_normal = runner.sdf_network.gradient(torch.tensor(vertices)).squeeze()

    view_dir = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    view_dir[:, 2] = 1
    print(view_dir)

    vertex_color = runner.color_network.forward(torch.tensor(vertices), surface_normal, torch.tensor(view_dir), appearance_feature).cpu().detach().numpy()
    print(vertex_color)
    print(vertices.shape, triangles.shape, vertex_color.shape)

    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_color)
    mesh.export(os.path.join(runner.base_exp_dir, 'meshes', 'vertex_color.ply'))

    logging.info('End')

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
    parser.add_argument('--validate_resolution', type=int, default=128)  # Higher value, clearer effect, 512
    # For rendering
    parser.add_argument('--render_resolution', type=float, default=4)  # Lower value, clearer effect
    parser.add_argument('--render_step', type=int, default=60)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)
    # visualize_intrinsic()

    set_vertex_color(runner, resolution=args.validate_resolution, threshold=args.mcube_threshold)
    # runner.funky_town(resolution_level=args.render_resolution, n_frames=args.render_step)
    # runner.validate_mesh(world_space=True, resolution=args.validate_resolution, threshold=args.mcube_threshold)
    # runner.interpolate_view(0, 26, resolution_level=args.render_resolution, n_frames=args.render_step)

    # Hi @will-thomson4, we do not need to generate rays to get a vertex color. Two networks are used in this work, one is the SDF network F, which takes the point coordinate as input, outputting the SDF value and the appearance feature vector; another is the rendering network G, which takes the appearance feature vector, the view direction and the surface normal as input, and outputs the color value. To get the color of a vertex, you may **first query the F network** to get the appearance feature and the surface normal (i.e., the gradient of SDF), manually assign an expected view direction, and feed them to the network G to get the vertex color.

    '''if args.mode == 'train':
        runner.train(resolution=args.train_resolution)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=args.validate_resolution, threshold=args.mcube_threshold)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1, resolution_level=args.render_resolution, n_frames=args.render_step)'''
