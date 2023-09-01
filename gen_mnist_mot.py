import os
import math
import cv2
import numpy as np
import torch
import sys
import shutil
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from random import randrange, uniform

# from joblib import Parallel, delayed

import paramparse

sys.path.append('../isl_labeling_tool/deep_mdp')

from utilities import col_bgr, annotate_and_show, SIIF, linux_path, draw_box
from gen_utils import rmdir, mkdir


class State:
    duration = 0
    patch = None
    bbox = None
    step = None
    target_id = None
    label = None


class Params:
    pause = 0
    # show_img = 1 shows vis images
    # show_img = 2 saves vis images
    show_img = 0

    batch_size = 20
    img_h = 128
    img_w = 0

    min_obj_size = 0
    max_obj_size = 0

    min_obj_h = 28
    min_obj_w = 0

    max_obj_h = 0
    max_obj_w = 0

    n_objs = 5
    # frame_num = 2.5e5
    # train_ratio = 0 if params.metric == 1 else 0.96
    # train_ratio = 0.6

    birth_prob = 0.5
    appear_interval = 5
    scale_var = 0
    ratio_var = 0

    velocity = 0
    min_velocity = 1.6
    max_velocity = 10.6

    task = 'mnist'
    out_dir = '/data'
    out_suffix = ''
    out_name = 'MNIST_MOT'
    # out_name = 'MNIST-MOT-512-56-112'
    eps = 1e-5
    rgb = 1

    save_fmt = 'mp4'
    codec = 'mp4v'
    fps = 30

    min_col_diff_percent = 25

    n_seq = 25
    n_frames = 2e3

    n_train_seq = 25
    n_test_seq = 25

    n_train_frames = 2e3
    n_test_frames = 2e3
    # train_frame_num = 20
    # test_frame_num = 20

    n_proc = 1


def get_col_diff(_col1, _col2):
    _col1_num = np.asarray(col_bgr[_col1], dtype=np.float32)
    _col2_num = np.asarray(col_bgr[_col2], dtype=np.float32)
    _col_abs_diff = np.sum(np.fabs(_col1_num - _col2_num)) / 3.0
    _col_abs_diff_percent = (_col_abs_diff / 255.0) * 100.0

    return _col_abs_diff_percent


def generate_batch(params: Params,
                   states,
                   seq_id, batch_id,
                   img_dir, vis_img_dir,
                   save_as_vid,
                   video_out,
                   vis_video_out,
                   data_num,
                   mnist_image_data, mnist_gt_data,
                   out_gt_data, img_ids,
                   target_ids, obj_cols_str, first_img_id,
                   bkg_col_str, valid_frg_cols):
    pause = params.pause

    # obj_h = params.obj_h
    # obj_w = params.obj_w

    h_range = params.max_obj_h - params.min_obj_h
    w_range = params.max_obj_w - params.min_obj_w

    h_rand = randrange(h_range + 1)
    w_rand = randrange(w_range + 1)

    obj_h = params.min_obj_h + h_rand
    obj_w = params.min_obj_w + w_rand

    m = obj_h // 3
    # nonlocal oid, first_img_id

    if params.rgb:
        n_channels = 3
        bkg_col = col_bgr[bkg_col_str]
        # buffer_big_np = np.zeros((2, params.img_h + 2 * obj_h, params.img_w + 2 * obj_w, n_channels), dtype=np.uint8)
        # buffer_big_np[..., 0] = bkg_col[0]
        # buffer_big_np[..., 1] = bkg_col[1]
        # buffer_big_np[..., 2] = bkg_col[2]
        # buffer_big = torch.from_numpy(buffer_big_np)
    else:
        n_channels = 1
        bkg_col = 0

    """allow for initial empty frames before any objects are born"""
    buffer_size = params.batch_size * 5

    buffer_big = torch.ByteTensor(2, params.img_h + 2 * obj_h, params.img_w + 2 * obj_w, n_channels).zero_()
    org_seq = torch.ByteTensor(buffer_size, params.img_h, params.img_w, n_channels).zero_()

    # sample all the random variables
    unif = torch.rand(buffer_size, params.n_objs)
    data_id = torch.rand(buffer_size, params.n_objs).mul_(data_num).floor_().long()
    direction_id = torch.rand(buffer_size, params.n_objs).mul_(4).floor_().long()  # [0, 3]
    position_id = torch.rand(buffer_size, params.n_objs, 2).mul_(params.img_h - 2 * m).add_(
        m).floor_().long()  # [m, params.img_h-m-1]
    scales = torch.rand(buffer_size, params.n_objs).mul_(2).add_(-1).mul_(params.scale_var).add_(
        1)  # [1 - var, 1 + var]
    ratios = torch.rand(buffer_size, params.n_objs).mul_(2).add_(-1).mul_(params.ratio_var).add_(
        1).sqrt_()  # [sqrt(1 - var), sqrt(1 + var)]

    batch_img_id = -1
    n_out_imgs = 0

    # img_dict = {}
    # vis_img_dict = {}

    while n_out_imgs < params.batch_size:

        batch_img_id += 1
        img_gt_data = []

        for obj_id in range(0, params.n_objs):
            obj_state = states[obj_id]
            if obj_state.duration < params.appear_interval:  # wait for interval frames
                obj_state.duration += 1
            elif obj_state.duration == params.appear_interval:  # allow birth
                if unif[batch_img_id][obj_id].item() < params.birth_prob:  # birth
                    # shape and appearance
                    data_ind = data_id[batch_img_id][obj_id].item()
                    scale = scales[batch_img_id][obj_id].item()
                    ratio = ratios[batch_img_id][obj_id].item()
                    patch_h, patch_w = round(obj_h * scale * ratio), round(obj_w * scale / ratio)
                    data_patch = mnist_image_data[data_ind]
                    data_label = int(mnist_gt_data[data_ind])
                    # data_patch = utils.imresize(data[data_ind], patch_h, patch_w).unsqueeze(2)
                    # pose
                    direction = direction_id[batch_img_id][obj_id].item()
                    position = position_id[batch_img_id][obj_id]
                    x1, y1, x2, y2 = None, None, None, None
                    if direction == 0:
                        x1 = position[0].item()
                        y1 = m
                        x2 = position[1].item()
                        y2 = params.img_h - 1 - m
                    elif direction == 1:
                        x1 = position[0].item()
                        y1 = params.img_h - 1 - m
                        x2 = position[1].item()
                        y2 = m
                    elif direction == 2:
                        x1 = m
                        y1 = position[0].item()
                        x2 = params.img_w - 1 - m
                        y2 = position[1].item()
                    else:
                        x1 = params.img_w - 1 - m
                        y1 = position[0].item()
                        x2 = m
                        y2 = position[1].item()
                    theta = math.atan2(y2 - y1, x2 - x1)

                    if params.velocity:
                        velocity = params.velocity
                    else:
                        velocity = uniform(params.min_velocity, params.max_velocity)

                    vx = velocity * math.cos(theta)
                    vy = velocity * math.sin(theta)

                    max_target_id = max(target_ids)
                    target_id = max_target_id + 1
                    target_ids.append(target_id)

                    # initial states
                    obj_state.duration = params.appear_interval + 1
                    obj_state.patch = data_patch
                    obj_state.bbox = [x1, y1, vx, vy]
                    obj_state.step = 0

                    obj_state.target_id = target_id
                    obj_state.label = data_label


            else:  # exists
                data_patch = obj_state.patch
                x1, y1, vx, vy = obj_state.bbox
                step = obj_state.step
                _target_id = obj_state.target_id + 1
                _obj_label = obj_state.label

                x = round(x1 + step * vx)
                y = round(y1 + step * vy)
                if x < m - params.eps or x > params.img_w - 1 - m + params.eps or y < m - params.eps or y > \
                        params.img_h - 1 - m + params.eps:  # the object disappears
                    obj_state.duration = 0
                else:
                    patch_h, patch_w = data_patch.size(0), data_patch.size(1)
                    # center and start position for the big image
                    center_x = x + obj_w
                    center_y = y + obj_h
                    top = math.floor(center_y - (patch_h - 1) / 2)
                    left = math.floor(center_x - (patch_w - 1) / 2)
                    # put the patch on image and synthesize a new frame
                    if params.rgb:

                        # data_patch_r = data_patch.detach().clone().squeeze()
                        # data_patch_g = data_patch.detach().clone().squeeze()
                        # data_patch_b = data_patch.detach().clone().squeeze()
                        try:
                            obj_col_str = obj_cols_str[_target_id]
                        except KeyError:
                            obj_col_id = np.random.randint(0, len(valid_frg_cols))
                            obj_cols_str[_target_id] = valid_frg_cols[obj_col_id]
                            obj_col_str = obj_cols_str[_target_id]

                        _col_abs_diff_percent = get_col_diff(bkg_col_str, obj_col_str)

                        assert _col_abs_diff_percent > params.min_col_diff_percent, \
                            f"invalid color combination: {bkg_col_str}, {obj_col_str}"

                        obj_col = col_bgr[obj_col_str]

                        # data_patch_r[data_patch_r != 0] = obj_col[0]
                        # data_patch_r[data_patch_r == 0] = bkg_col[0]

                        # data_patch_g[data_patch_g != 0] = obj_col[1]
                        # data_patch_g[data_patch_g == 0] = bkg_col[1]

                        # data_patch_b[data_patch_b != 0] = obj_col[2]
                        # data_patch_b[data_patch_b == 0] = bkg_col[2]

                        data_patch_rgb = torch.stack((data_patch, data_patch, data_patch), dim=2).squeeze()

                        img = buffer_big[0].detach().clone()
                        img.narrow(0, top, patch_h).narrow(1, left, patch_w).copy_(data_patch_rgb)
                        img = img.narrow(0, obj_h, params.img_h).narrow(1, obj_w, params.img_w)

                        # if params.show_img:
                        # data_patch_np = data_patch.detach().numpy().squeeze()
                        # data_patch_rgb_np = data_patch_rgb.detach().numpy().squeeze()
                        # annotate_and_show('data_patch_gs', data_patch_np)
                        # img_np = img.detach().numpy().squeeze()
                        # annotate_and_show('img_np', img_np)

                        img_gs = img[..., 0].squeeze()
                        obj_pixels = (img_gs != 0)
                        # bkg_pixels = (data_patch == 0)

                        img_f = img.float()

                        """extract the existing current image to add this object to that image"""
                        org_img_f = org_seq[batch_img_id].float()  # params.img_h * params.img_w * params.n_channels
                        syn_image_gs = (org_img_f + img_f).clamp_(max=255)

                        syn_image = syn_image_gs.detach().clone()

                        # syn_image[..., 0][obj_pixels] = img_gs[obj_pixels]*(obj_col[0]/255.0)
                        # syn_image[..., 1][obj_pixels] = img_gs[obj_pixels]*(obj_col[1]/255.0)
                        # syn_image[..., 2][obj_pixels] = img_gs[obj_pixels]*(obj_col[2]/255.0)

                        syn_image[..., 0][obj_pixels] = obj_col[0]
                        syn_image[..., 1][obj_pixels] = obj_col[1]
                        syn_image[..., 2][obj_pixels] = obj_col[2]

                        # syn_image_gs_np = syn_image_gs.detach().numpy().squeeze().astype(np.uint8)
                        # syn_image_np = syn_image.detach().numpy().squeeze().astype(np.uint8)
                        # syn_image_gs_np_r = syn_image_gs_np[..., 0].squeeze()
                        # syn_image_np_r = syn_image_np[..., 0].squeeze()
                        # annotate_and_show('syn_image_np', syn_image_np)

                        # syn_image = img_f.clamp_(max=255)
                    else:
                        img = buffer_big[0].zero_()
                        img.narrow(0, top, patch_h).narrow(1, left, patch_w).copy_(data_patch)
                        img = img.narrow(0, obj_h, params.img_h).narrow(1, obj_w, params.img_w)
                        img_f = img.float()
                        """extract the existing current image to add this object to that image"""
                        org_img_f = org_seq[batch_img_id].float()
                        syn_image = (org_img_f + img_f).clamp_(max=255)

                        # print()

                    """put back the current image with the current object added onto it"""
                    org_seq[batch_img_id].copy_(syn_image.round().byte())

                    # update the position
                    obj_state.step += 1

                    _obj_gt_data = [_target_id, left - obj_w + 1, top - obj_h + 1, patch_w, patch_h, _obj_label]

                    if params.rgb:
                        obj_col_str = obj_cols_str[_target_id]
                        obj_col = col_bgr[obj_col_str][::-1]
                        obj_col_num_str = '_'.join(map(str, obj_col))

                        _obj_gt_data += [obj_col_str, obj_col_num_str]

                    img_gt_data.append(_obj_gt_data)

                    # print('batch_id: {} out_img_id: {}'.format(batch_id, out_img_id))

        if img_gt_data:

            out_img_id = batch_id * params.batch_size + n_out_imgs + 1
            if first_img_id is None:
                first_img_id = out_img_id - 1

            out_img_id -= first_img_id

            # print('output_input_dir: {}'.format(output_input_dir))
            # print('out_path: {}'.format(out_path))
            img_ids.append(out_img_id)

            torch_img = org_seq[batch_img_id]
            img = torch_img.cpu().numpy().copy()

            out_gt_data[out_img_id] = img_gt_data

            if params.rgb:
                """add background color"""
                img_r = img[..., 0]
                img_g = img[..., 1]
                img_b = img[..., 2]
                img_r[img_r == 0] = bkg_col[0]
                img_g[img_g == 0] = bkg_col[1]
                img_b[img_b == 0] = bkg_col[2]

            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out_fname = 'image{:06d}.jpg'.format(out_img_id)

            if save_as_vid:
                video_out.write(img)
            else:
                out_path = linux_path(img_dir, out_fname)
                cv2.imwrite(out_path, img)

            if params.show_img:
                vis_img = img.copy()

                for _obj_gt_data in img_gt_data:
                    _target_id, xmin, ymin, obj_w, obj_h, label = _obj_gt_data[:6]

                    bbox = [xmin, ymin, obj_w, obj_h]

                    draw_box(vis_img, bbox, f'{_target_id}')

                if params.show_img == 2:
                    if save_as_vid:
                        vis_video_out.write(vis_img)
                    else:
                        vis_out_path = linux_path(vis_img_dir, out_fname)
                        cv2.imwrite(vis_out_path, vis_img)
                else:
                    vis_img_show = annotate_and_show('vis_img', vis_img,
                                                     text=f'seq {seq_id} batch {batch_id} frame {out_img_id}',
                                                     n_modules=0, only_annotate=1)
                    cv2.imshow('vis_img', vis_img_show)
                    k = cv2.waitKey(1 - pause)
                    if k == 32:
                        pause = 1 - pause
                    elif k == 27:
                        exit()

            n_out_imgs += 1

    return org_seq, states, first_img_id


def generate_seq(
        seq_info,
        n_seq,
        n_cols,
        all_cols,
        valid_cols,
        save_as_vid,
        batch_nums,
        data_num,
        mnist_image_data,
        mnist_gt_data,
        output_img_dir,
        output_gt_dir,
        output_vis_dir,
        params: Params
):
    split, seq_id = seq_info

    n_dig = len(str(n_seq[split]))
    fmt = f'%0{n_dig}d'
    seq_id_str = fmt % seq_id

    print(f'\n{split} seq {seq_id + 1} / {n_seq[split]}\n')

    target_ids = []
    obj_cols_str = {}
    first_img_id = None
    states = [State() for _ in range(params.n_objs)]

    seq_name = f'{split}_{seq_id_str}'

    bkg_col_str = None
    valid_frg_cols = None

    if params.rgb:
        bkg_col_id = np.random.randint(0, n_cols)
        bkg_col_str = all_cols[bkg_col_id]
        bkg_col = col_bgr[bkg_col_str][::-1]
        bkg_col_num_str = '_'.join(map(str, bkg_col))

        # valid_frg_cols = all_cols[:]
        # valid_frg_cols.remove(bkg_col_str)
        valid_frg_cols = valid_cols[bkg_col_str]

        assert valid_frg_cols, "no valid_frg_cols found for {}".format(bkg_col_str)

        seq_name = f'{seq_name}_{bkg_col_str}_{bkg_col_num_str}'

        print('bkg_col_str: {}'.format(bkg_col_str))
        print('bkg_col_num_str: {}'.format(bkg_col_num_str))
        print('n_valid_frg_cols: {}'.format(len(valid_frg_cols)))

    video_out = None
    vis_video_out = None

    mot_gt_path = linux_path(output_gt_dir, f'{seq_name}.txt')

    if save_as_vid:
        codec = params.codec
        fourcc = cv2.VideoWriter_fourcc(*codec)

        img_path = linux_path(output_img_dir, f'{seq_name}.{params.save_fmt}')
        video_out = cv2.VideoWriter(img_path, fourcc, params.fps, (params.img_w, params.img_h))

        vis_img_path = linux_path(output_vis_dir, f'{seq_name}.{params.save_fmt}')
        csv_file_path = linux_path(output_img_dir, f'{seq_name}.csv')

        if params.show_img == 2:
            vis_video_out = cv2.VideoWriter(vis_img_path, fourcc, params.fps, (params.img_w, params.img_h))
    else:
        img_path = linux_path(output_img_dir, seq_name)
        vis_img_path = linux_path(output_vis_dir, seq_name)
        os.makedirs(img_path, exist_ok=1)
        csv_file_path = linux_path(img_path, 'annotations.csv')
        if params.show_img == 2:
            os.makedirs(vis_img_path, exist_ok=1)

    print(f'saving images to {img_path}')
    print(f'saving csv GT to {csv_file_path}')
    print(f'saving MOT GT to {mot_gt_path}')

    # img_id = 0
    img_ids = []
    gt_data = defaultdict(list)

    n_batches = batch_nums[split]
    for batch_id in tqdm(range(n_batches), ncols=80):  # for each batch of images
        out_batch = generate_batch(
            params, states,
            seq_id, batch_id,
            img_path,
            vis_img_path,
            save_as_vid,
            video_out,
            vis_video_out,
            data_num, mnist_image_data, mnist_gt_data,
            gt_data, img_ids,
            target_ids, obj_cols_str, first_img_id,
            bkg_col_str, valid_frg_cols)

        org_seq_batch, states, first_img_id = out_batch

        # print(seq_name + ': ' + str(batch_id + 1) + ' / ' + str(n_batches))

    print('fixing missing image IDs')
    img_ids_unique = list(set(img_ids))
    img_ids_sorted = sorted(img_ids_unique)
    mot_gt_file = open(mot_gt_path, "w")
    csv_raw = []

    n_frames = len(img_ids_sorted)

    for i, src_img_id in enumerate(img_ids_sorted):
        dst_img_id = i + 1

        src_fname = f'image{src_img_id:06d}.jpg'
        dst_fname = f'image{dst_img_id:06d}.jpg'

        for _gt_data in gt_data[src_img_id]:
            _target_id, xmin, ymin, obj_w, obj_h, label = _gt_data[:6]

            mot_gt_file.write("{:d},{:d},{:.3f},{:.3f},{:.3f},{:.3f},1,-1,-1,-1\n".format(
                dst_img_id, _target_id, xmin, ymin, obj_w, obj_h
            ))

            xmax = xmin + obj_w
            ymax = ymin + obj_h

            raw_data = {
                'filename': dst_fname,
                'width': int(params.img_w),
                'height': int(params.img_h),
                'class': label,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax,
                'confidence': 1.0,
                'target_id': _target_id,
            }

            if params.rgb:
                obj_col, obj_col_num = _gt_data[6:8]
                raw_data.update(
                    {
                        'color': obj_col,
                        'color_num': obj_col_num,
                    }
                )

            csv_raw.append(raw_data)

        if src_img_id == dst_img_id:
            continue

        if save_as_vid:
            raise AssertionError("mismatch between source and destination image IDs")

        src_path = linux_path(img_path, src_fname)
        dst_path = linux_path(img_path, dst_fname)
        shutil.move(src_path, dst_path)

    mot_gt_file.close()
    if video_out is not None:
        video_out.release()
    if vis_video_out is not None:
        vis_video_out.release()

    df = pd.DataFrame(csv_raw)
    df.to_csv(csv_file_path)

    return seq_name, n_frames


def main():
    params = Params()
    paramparse.process(params)

    vid_exts = ['mkv', 'mp4', 'avi', 'mjpg', 'wmv']
    image_exts = ['jpg', 'bmp', 'png', 'tif']

    if params.save_fmt in vid_exts:
        save_as_vid = 1
    elif params.save_fmt in image_exts:
        save_as_vid = 0
    else:
        raise AssertionError(f'invalid save format: {params.save_fmt}')

    # N = 1 if params.metric == 1 else 64
    # N = 1

    params.n_frames = int(params.n_frames)

    if params.img_w == 0:
        params.img_w = params.img_h

    if params.min_obj_size:
        params.min_obj_w = params.min_obj_h = params.min_obj_size

    if params.max_obj_size:
        params.max_obj_w = params.max_obj_h = params.max_obj_size

    if params.min_obj_w == 0:
        params.min_obj_w = params.min_obj_h

    if params.max_obj_h == 0:
        params.max_obj_h = params.min_obj_h

    if params.max_obj_w == 0:
        params.max_obj_w = params.min_obj_w

    # txt_name = 'gt.txt'
    # metric_dir = 'metric' if params.metric == 1 else ''
    data_dir = linux_path('data', params.task)
    input_dir = linux_path(data_dir, 'MNIST', 'processed')
    out_name = params.out_name
    if params.rgb:
        out_name = f'{out_name}_RGB'
    out_name = f'{out_name}_{params.img_h}x{params.img_w}_{params.n_objs}_{params.n_seq}_{params.n_frames}'

    if not params.velocity:
        out_name = f'{out_name}_var'

    if params.out_suffix:
        out_name = f'{out_name}_{params.out_suffix}'

    output_dir = linux_path(params.out_dir, out_name)

    print(f'output_dir: {output_dir}')

    output_img_dir = linux_path(output_dir, 'Images')
    output_gt_dir = linux_path(output_dir, 'Annotations')
    output_vis_dir = linux_path(output_dir, 'vis')

    output_img_dir = os.path.abspath(output_img_dir)
    output_gt_dir = os.path.abspath(output_gt_dir)
    output_vis_dir = os.path.abspath(output_vis_dir)

    rmdir(output_img_dir)
    mkdir(output_img_dir)

    rmdir(output_gt_dir)
    mkdir(output_gt_dir)

    rmdir(output_vis_dir)
    mkdir(output_vis_dir)

    # mnist data
    if not os.path.exists(input_dir):
        import torchvision.datasets as datasets

        datasets.MNIST(root=data_dir, train=True, download=True)
    train_data = torch.load(linux_path(input_dir, 'training.pt'))  # 60000 * 28 * 28
    test_data = torch.load(linux_path(input_dir, 'test.pt'))  # 10000 * 28 * 28
    mnist_image_data = torch.cat((train_data[0], test_data[0]), 0).unsqueeze(3)  # 70000 * 28 * 28
    data_num = mnist_image_data.size(0)
    mnist_gt_data = torch.cat((train_data[1], test_data[1]))  # 70000 * 1

    # generate data from trackers

    if params.n_seq > 0:
        params.n_train_seq = params.n_test_seq = params.n_seq

    if params.n_frames > 0:
        params.n_train_frames = params.n_test_frames = params.n_frames

    assert params.batch_size <= params.n_train_frames, "batch_size cannot be greater than the number of training frames"
    assert params.batch_size <= params.n_test_frames, "batch_size cannot be greater than the number of test frames"

    print('train frame number: ' + str(params.n_train_frames))
    print('test frame number: ' + str(params.n_test_frames))
    batch_nums = {
        'train': math.floor(params.n_train_frames / params.batch_size),
        'test': math.floor(params.n_test_frames / params.batch_size)
    }
    n_seq = {
        'train': params.n_train_seq,
        'test': params.n_test_seq
    }

    # core_num = 1 if params.metric == 1 else multiprocessing.cpu_count()
    # core_num = 1
    # print("Running with " + str(core_num) + " cores.")

    all_cols = list(col_bgr.keys())
    """0 pixels used to distinguish object and non-object pixels before coloring images"""
    all_cols.remove('black')
    # all_cols = ['green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
    #             'dark_slate_gray', 'navy', 'turquoise', 'white']
    n_cols = len(all_cols)

    all_cols.sort()

    all_cols_num_str = ['_'.join(map(str, col_bgr[_col])) for _col in all_cols]

    col_diffs = pd.DataFrame(
        np.full((len(all_cols), len(all_cols)), 0, dtype=np.float32),
        columns=all_cols, index=all_cols)

    valid_cols = {
        _col: []
        for _col in all_cols
    }
    valid_cols_dbg = {
        (_col, col_bgr[_col]): []
        for _col in all_cols
    }
    for _id, _col1 in enumerate(all_cols):
        for _col2 in all_cols[_id + 1:]:
            _col_abs_diff_percent = get_col_diff(_col1, _col2)
            col_diffs[_col1][_col2] = _col_abs_diff_percent
            col_diffs[_col2][_col1] = _col_abs_diff_percent

            if _col_abs_diff_percent > params.min_col_diff_percent:
                valid_cols[_col2].append(_col1)
                valid_cols[_col1].append(_col2)

                _col1_num = col_bgr[_col1]
                _col2_num = col_bgr[_col2]

                valid_cols_dbg[(_col2, _col2_num)].append((_col1, _col1_num, _col_abs_diff_percent))
                valid_cols_dbg[(_col1, _col1_num)].append((_col2, _col2_num, _col_abs_diff_percent))

    bkg_col_str = ''

    n_valid_cols = {
        _col: len(valid_cols[_col])
        for _col in all_cols
    }
    n_valid_cols_list = list(n_valid_cols.values())

    min_valid_cols = np.amin(n_valid_cols_list)
    max_valid_cols = np.amax(n_valid_cols_list)

    assert min_valid_cols > 50, "min_valid_cols is too low"

    seq_names = []
    seq_n_frames = []
    valid_frg_cols = None

    seq_info_list = [(split, seq_id) for split in ['train', 'test'] for seq_id in range(n_seq[split])]

    import functools

    func = functools.partial(
        generate_seq,
        n_seq=n_seq,
        n_cols=n_cols,
        all_cols=all_cols,
        valid_cols=valid_cols,
        save_as_vid=save_as_vid,
        batch_nums=batch_nums,
        data_num=data_num,
        mnist_image_data=mnist_image_data,
        mnist_gt_data=mnist_gt_data,
        output_img_dir=output_img_dir,
        output_gt_dir=output_gt_dir,
        output_vis_dir=output_vis_dir,
        params=params,
    )

    n_proc = params.n_proc
    if n_proc > 1:
        import multiprocessing

        print(f'running in parallel over {n_proc} processes')
        with multiprocessing.Pool(n_proc) as pool:
            results = pool.map(func, seq_info_list)

        results.sort(key=lambda x: x[0])
    else:
        results = []
        for seq_info in seq_info_list:
            result = func(seq_info)

            results.append(result)

    seq_names, seq_n_frames = list(zip(*results))
    n_seq = len(seq_names)
    seq_names_list_str = ',\n'.join(map(
        lambda _seq_id: f"{_seq_id}: ('{seq_names[_seq_id]}', {seq_n_frames[_seq_id]})",
        range(n_seq)))

    seq_info_path = linux_path(output_dir, 'seq_info.txt')
    print('saving seq_info  to {}'.format(seq_info_path))

    with open(seq_info_path, "w") as fid:
        fid.write(seq_names_list_str)


if __name__ == '__main__':
    SIIF.setup()

    main()
