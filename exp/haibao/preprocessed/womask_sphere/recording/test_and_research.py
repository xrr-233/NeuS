import os
import logging
import argparse

import cv2
import trimesh

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def set_vertex_color(runner, resolution, threshold):  # 失败
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

def export_o_v_point_cloud(runner, resolution, threshold, id):
    rays_o, rays_v = runner.dataset.gen_rays_at(id, resolution_level=16)
    H, W, _ = rays_o.shape

    rays_o = rays_o.cpu().numpy().reshape(-1, 3)[0].reshape(1, 3)
    # 分步理解rays_v
    # l = 16
    # tx = torch.linspace(0, runner.dataset.W - 1, runner.dataset.W // l)
    # ty = torch.linspace(0, runner.dataset.H - 1, runner.dataset.H // l)
    # pixels_x, pixels_y = torch.meshgrid(tx, ty)
    # rays_v = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
    # rays_v = torch.matmul(runner.dataset.intrinsics_all_inv[id, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # 与内参矩阵相乘，W, H, 3
    # rays_v = rays_v / torch.linalg.norm(rays_v, ord=2, dim=-1, keepdim=True)  # 求L2范式，W, H, 3
    # rays_v = torch.matmul(runner.dataset.pose_all[id, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
    rays_v = rays_v.cpu().numpy().reshape(-1, 3)

    bound_min = torch.tensor(runner.dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(runner.dataset.object_bbox_max, dtype=torch.float32)
    vertices, triangles = runner.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
    # vertices = np.array(vertices * runner.dataset.scale_mats_np[id][0, 0] + runner.dataset.scale_mats_np[id][:3, 3][None], dtype=np.float32)

    print(rays_o.shape)
    print(rays_v.shape)
    print(vertices.shape)
    print(np.min(vertices, axis=0))
    print(np.max(vertices, axis=0))

    res = np.concatenate((rays_o[0].reshape(-1, 3), rays_v, vertices), axis=0)
    print(res.shape)
    with open('points.txt', 'w') as f:
        f.write(f"{rays_o.shape[0]} {rays_v.shape[0]} {vertices.shape[0]} {H} {W} \n")
        for i in tqdm(range(res.shape[0])):
            for j in range(res.shape[1]):
                f.write(f"{res[i][j]} ")
            f.write('\n')

def export_projection(runner, resolution, threshold, id):
    bound_min = torch.tensor(runner.dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(runner.dataset.object_bbox_max, dtype=torch.float32)

    vertices, triangles = runner.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
    vertices = np.array(vertices * runner.dataset.scale_mats_np[0][0, 0] + runner.dataset.scale_mats_np[0][:3, 3][None], dtype=np.float32)

    world_point = np.ones((vertices.shape[0], 4))
    world_point[:, :3] = vertices

    print(runner.dataset.world_mats_np[0])
    print(runner.dataset.scale_mats_np[0])

    P = (runner.dataset.world_mats_np[0] @ runner.dataset.scale_mats_np[0])[:3, :4]

    print(world_point[0])
    print(P)
    print(runner.dataset.intrinsics_all[0])
    print(runner.dataset.pose_all[0])
    image_point = (P @ world_point.T).T
    print(image_point[0])

    with open('points2.txt', 'w') as f:
        for i in tqdm(range(image_point.shape[0])):
            for j in range(image_point.shape[1]):
                f.write(f"{image_point[i][j]} ")
            f.write('\n')

def validate_image(runner, rays_o, rays_d, filename='validations_fine.png', generate_mask=False):
    print("Validating image")
    H, W, _ = rays_o.shape
    rays_o = rays_o.reshape(-1, 3).split(runner.batch_size)
    rays_d = rays_d.reshape(-1, 3).split(runner.batch_size)

    out_rgb_fine = []

    for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)): # 94 for 16, 23814 for 1
        near, far = runner.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
        background_rgb = torch.ones([1, 3]) if runner.use_white_bkgd else None

        render_out = runner.renderer.render(rays_o_batch,
                                          rays_d_batch,
                                          near,
                                          far,
                                          cos_anneal_ratio=runner.get_cos_anneal_ratio(),
                                          background_rgb=background_rgb,
                                          generate_mask=generate_mask)

        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
        del render_out

    img_fine = None
    if len(out_rgb_fine) > 0:
        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255)
    print(img_fine.shape)
    cv2.imwrite(filename, img_fine)
    print("Validation end")

def brute_force(runner, resolution, threshold, id):
    rays_o, rays_v = runner.dataset.gen_rays_at_no_normalization(id, resolution_level=16)
    H, W, _ = rays_o.shape
    # validate_image(runner, rays_o, rays_v)

    rays_o = rays_o.cpu().numpy().reshape(-1, 3)
    rays_v = rays_v.cpu().numpy().reshape(-1, 3)
    bound_min = torch.tensor(runner.dataset.object_bbox_min, dtype=torch.float32)
    bound_max = torch.tensor(runner.dataset.object_bbox_max, dtype=torch.float32)
    vertices, triangles = runner.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
    # vertices = np.array(vertices * runner.dataset.scale_mats_np[id][0, 0] + runner.dataset.scale_mats_np[id][:3, 3][None], dtype=np.float32)
    # vertices = np.array(vertices + runner.dataset.scale_mats_np[id][:3, 3][None], dtype=np.float32)
    print(runner.dataset.scale_mats_np[id])
    print(rays_o[0])
    rays_o_scale = rays_o
    rays_v_scale = rays_v
    # rays_o_scale = np.array(rays_o * runner.dataset.scale_mats_np[id][0, 0] + runner.dataset.scale_mats_np[id][:3, 3][None], dtype=np.float32)
    # rays_v_scale = np.array(rays_v * runner.dataset.scale_mats_np[id][0, 0] + runner.dataset.scale_mats_np[id][:3, 3][None], dtype=np.float32)
    print(rays_o_scale[0])

    # print(rays_o.shape)
    # print(rays_v.shape)
    # print(vertices.shape)
    print(np.min(vertices, axis=0))
    print(np.max(vertices, axis=0))

    res = np.concatenate((rays_o_scale[0].reshape(-1, 3), rays_v_scale, vertices), axis=0)
    # print(res.shape)
    print("Writing points.txt")
    with open('points.txt', 'w') as f:
        f.write(f"1 {rays_v.shape[0]} {vertices.shape[0]} {H} {W} \n")
        for i in tqdm(range(res.shape[0])):
            for j in range(res.shape[1]):
                f.write(f"{res[i][j]} ")
            f.write('\n')

    print("Giving vertex color")
    rays_d = np.zeros(vertices.shape, dtype=np.float32)
    thetas = np.zeros(vertices.shape[0], dtype=np.float32)
    for i in tqdm(range(vertices.shape[0])):
        oi = vertices[i] - rays_o_scale[0]
        oj = rays_v_scale - rays_o_scale
        theta = np.arccos(np.sum(oi * oj, axis=1) / np.linalg.norm(oi) / np.linalg.norm(oj, axis=1))
        theta_ray = np.argmin(theta)
        thetas[i] = theta_ray
        rays_d[i] = rays_v[theta_ray]
        # if (i == 0):
        #     print(oi)
        #     print(oj[:10])
        #     print(theta[:10])
        #     print(theta_ray)
    print(thetas[:10])

    rays_o = np.broadcast_to(rays_o[0], rays_d.shape)

    rays_o = torch.tensor(rays_o).split(runner.batch_size)
    rays_d = torch.tensor(rays_d).split(runner.batch_size)
    out_rgb_fine = []
    for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
        near, far = runner.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
        background_rgb = torch.ones([1, 3]) if runner.use_white_bkgd else None

        render_out = runner.renderer.render(rays_o_batch,
                                          rays_d_batch,
                                          near,
                                          far,
                                          cos_anneal_ratio=runner.get_cos_anneal_ratio(),
                                          background_rgb=background_rgb)

        out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

        del render_out

    img_fine = np.concatenate(out_rgb_fine, axis=0).reshape((-1, 3))
    img_fine = cv2.cvtColor(img_fine[:, None, :], cv2.COLOR_RGB2BGR).reshape((-1, 3))

    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=img_fine)
    mesh.export(os.path.join(runner.base_exp_dir, 'meshes', 'vertex_color.ply'))

def generate_mask(runner, generate_mask=True):
    img_num = runner.dataset.n_images
    os.makedirs('masks', exist_ok=True)

    for idx in range(img_num):
        print(idx)
        rays_o, rays_v = runner.dataset.gen_rays_at(idx, resolution_level=1)
        validate_image(runner, rays_o, rays_v, filename=os.path.join('masks', '%03d.png' % idx), generate_mask=generate_mask)

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
    # set_vertex_color(runner, resolution=args.validate_resolution, threshold=args.mcube_threshold)
    # export_o_v_point_cloud(runner, resolution=args.validate_resolution, threshold=args.mcube_threshold, id=0)
    # export_projection(runner, resolution=args.validate_resolution, threshold=args.mcube_threshold, id=14)
    brute_force(runner, resolution=args.validate_resolution, threshold=args.mcube_threshold, id=14)
    # runner.funky_town(resolution_level=args.render_resolution, n_frames=args.render_step)
    # generate_mask(runner, generate_mask=False)
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
