import os
os.environ['PATH'] = "/home/bsft19/ruixu33/Programs/anaconda3/bin:/home/bsft19/ruixu33/Programs/anaconda3/condabin:/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/usr/games:/usr/local/games:/home/bsft19/ruixu33/Programs/anaconda3/bin"

import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm
from models.runner import Runner

runner = None


def validate_image(rays_o, rays_d, filename='validations_fine.png', generate_mask=False):
    print("Validating image")
    H, W, _ = rays_o.shape
    rays_o_batch = rays_o.reshape(-1, 3).split(runner.batch_size)
    rays_d_batch = rays_d.reshape(-1, 3).split(runner.batch_size)
    render_iter = len(rays_o)

    out_rgb_fine = []

    for iter in tqdm(range(render_iter)):  # 94 for 16, 23814 for 1
        near, far = runner.dataset.near_far_from_sphere(rays_o_batch[iter], rays_d_batch[iter])
        background_rgb = torch.ones([1, 3]) if runner.use_white_bkgd else None

        render_out = runner.renderer.render(rays_o_batch[iter],
                                            rays_d_batch[iter],
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


def generate_mask():
    img_num = runner.dataset.n_images
    os.makedirs('masks', exist_ok=True)

    for idx in range(img_num):
        print(idx)
        rays_o, rays_v = runner.dataset.gen_rays_at(idx, resolution_level=1)
        validate_image(rays_o, rays_v, filename=os.path.join('masks', '%03d.png' % idx), generate_mask=True)


if __name__ == '__main__':
    print('Hello Wooden')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    # logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/womask.conf')
    parser.add_argument('--mode', type=str, default='validate_mesh')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str)  # Please specify your dataset folder in the 'exp' folder

    parser.add_argument('--train_resolution', type=int, default=64)
    parser.add_argument('--validate_resolution', type=int, default=512)  # Higher value, clearer effect, default=512
    # For rendering
    parser.add_argument('--render_resolution', type=float, default=4)  # Lower value, clearer effect, default=4
    parser.add_argument('--render_step', type=int, default=60)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train(resolution=args.train_resolution, final_resolution=args.validate_resolution, threshold=args.mcube_threshold)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh_vertex_color(world_space=True, resolution=args.validate_resolution, threshold=args.mcube_threshold, name='self_defined_result')
    elif args.mode == 'generate_mask':
        generate_mask()
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1, resolution_level=args.render_resolution, n_frames=args.render_step)
