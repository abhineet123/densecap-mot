import numpy as np
import os
import sys
import cv2
from datetime import datetime
from tqdm import tqdm
import math
import glob

sys.path.append('../isl_labeling_tool/deep_mdp')

from utilities import draw_box, show, annotate_and_show, compute_overlap, prob_to_rgb2, draw_traj2, resize_ar, \
    linux_path, CVText

from objects import Annotations


def get_latest_checkpoint(dir_name, prefix='epoch_', ignore_missing=False):
    ckpt_names = glob.glob(f'{dir_name}/{prefix}*.pth')

    if len(ckpt_names) == 0:
        msg = f'No checkpoints found in {dir_name}'
        if ignore_missing:
            print(msg)
            return None, None
        raise AssertionError(msg)

    ckpt_names.sort(key=lambda x: os.path.getmtime(x))
    checkpoint = ckpt_names[-1]

    checkpoint_name = os.path.splitext(os.path.basename(checkpoint))[0]

    epoch_str = checkpoint_name.replace(prefix, '')

    try:
        epoch = int(epoch_str)
    except ValueError:
        raise AssertionError(f'invalid checkpoint_name: {checkpoint_name} for prefix: {prefix}')

    print(f'latest checkpoint found from epoch {epoch}:  {checkpoint}')

    return checkpoint, epoch


def build_targets_densecap(
        vocab_fmt: int,
        max_diff: int,
        sample_traj: int,
        n_frames: int,
        frame_size: tuple,
        frames: list,
        annotations: Annotations,
        seq_name,
        grid_res,
        frame_gap,
        win_size,
        fps,
        out_dir,
        vis):
    if vis:
        assert frames is not None, "frames must be provided for visualization"

    if win_size <= 0:
        print('Temporal Windows are disabled')
        win_size = n_frames

    last_start_frame_id = n_frames - win_size
    grid_cell_size = np.array([frame_size[i] / grid_res[i] for i in range(2)])

    """grid cell centers
    """
    grid_x, grid_y = [np.arange(grid_cell_size[i] / 2.0, frame_size[i], grid_cell_size[i]) for i in range(2)]
    grid_cx, grid_cy = np.meshgrid(grid_x, grid_y)

    n_grid_cells = grid_cx.size

    ann_sizes = annotations.data[:, 4:6]
    ann_min = annotations.data[:, 2:4]
    ann_centers = ann_min + ann_sizes / 2.0

    obj_cols = (
        'forest_green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
        'dark_slate_gray',
        'navy', 'turquoise'
    )

    n_obj_cols = len(obj_cols)
    vocab_annotations = [None] * annotations.n_traj
    vocab = []
    prev_grid_ids = [None] * annotations.n_traj
    traj_lengths = []

    max_traj_length = 0
    min_traj_length = np.inf

    for traj_id in range(annotations.n_traj):
        traj_frame_ids = list(annotations.traj_idx_by_frame[traj_id].keys())
        min_frame_id, max_frame_id = np.amin(traj_frame_ids), np.amax(traj_frame_ids)

        traj_length = max_frame_id - min_frame_id + 1

        traj_lengths.append(traj_length)

        vis_traj = 0

        if traj_length < min_traj_length:
            min_traj_length = traj_length
            vis_traj = 1

        if traj_length > max_traj_length:
            max_traj_length = traj_length
            vis_traj = 1

        # if vis_traj:
        #     obj_id = annotations.traj_to_obj[traj_id]
        #     if vis:
        #         codec = 'mp4v'
        #         fourcc = cv2.VideoWriter_fourcc(*codec)
        #         vis_out_dir = linux_path(out_dir, 'vis')
        #         os.makedirs(vis_out_dir, exist_ok=1)
        #         vis_out_path = linux_path(vis_out_dir, f'{seq_name}_{traj_id}_{traj_length}.mp4')
        #
        #         print(f'vis_out_path: {vis_out_path}')
        #
        #         img_h, img_w = frames[min_frame_id].shape[:2]
        #
        #         video_out = cv2.VideoWriter(vis_out_path, fourcc, fps, (img_w, img_h))
        #
        #         for _frame_id in range(min_frame_id, max_frame_id + 1):
        #             frame = frames[_frame_id]
        #             frame_disp = np.copy(frame)
        #             ann_idx = annotations.idx[_frame_id]
        #             frame_ann_data = annotations.data[ann_idx]
        #             curr_obj_data = [_data for _data in frame_ann_data if _data[1] == obj_id]
        #
        #             assert len(curr_obj_data) == 1, "something annoying going on"
        #
        #             curr_obj_data = curr_obj_data[0]
        #
        #             draw_box(frame_disp, curr_obj_data[2:6], color='green',
        #                      header=f'frame {_frame_id}', header_fmt=CVText(color='red'))
        #
        #             video_out.write(frame_disp)
        #
        #             cv2.imshow('frame_disp', frame_disp)
        #             cv2.waitKey(1)
        #
        #         video_out.release()

        # if traj_length > 200:
        #     print(f'excessive trajectory length: {traj_length}')

        # traj_length_to_count[traj_length] += 1

        min_time, max_time = float(min_frame_id) / fps, float(max_frame_id) / fps
        vocab_annotations[traj_id] = dict(
            segment=[min_time, max_time],
            id=traj_id,
            sentence='',
            # grid_cells=[],
        )

    out_img_id = 0

    _pause = 1

    win_iter = range(0, last_start_frame_id + 1, frame_gap)

    if len(win_iter) > 1:
        win_iter = tqdm(win_iter)

    """
    Iterate over all the temporal windows in the sequence
    Note that temporal windows are really only needed during inference
    For generating the training data, all the trajectories over the entire sequence 
    should be considered together
    """
    for win_id, start_frame_id in enumerate(win_iter):
        end_frame_id = min(start_frame_id + win_size, n_frames) - 1
        """
        Iterate over all the frames in this temporal window
        """

        n_frames = end_frame_id - start_frame_id + 1
        """sampling the entire sequence instead of each trajectory individually 
        can cause some trajectories
         to become severely truncated depending on What is the nearest sampled frame to the 
         last frame in the trajectory which might well be long before the actual frame
         However, if the purpose of this sort of sampling is to match the 
         trajectories with the available features in the similarly sampled feature extractor, 
         then this is indeed probably the best way to do this
         """
        if sample_traj > 1:
            n_frames = int(math.ceil(n_frames / sample_traj))
            frame_iter = list(np.linspace(start_frame_id, end_frame_id, n_frames, dtype=np.int32))
        else:
            frame_iter = range(start_frame_id, end_frame_id + 1)

        if len(win_iter) > 1:
            win_iter.set_description(f'win {win_id}: frame {start_frame_id} --> {end_frame_id}')
        else:
            frame_iter = tqdm(frame_iter)

        for frame_id in frame_iter:

            ann_idx = annotations.idx[frame_id]
            obj_centers = ann_centers[ann_idx, :]

            """Map each object to the grid that contains its centre
            """
            obj_grid_ids = (obj_centers.T / grid_cell_size[:, None]).astype(np.int64)
            # arr = np.array([[3, 6, 6], [4, 5, 1]])

            """np.ravel_multi_index takes row, col indices"""
            obj_grid_ids_flat = np.ravel_multi_index(obj_grid_ids[::-1], grid_res)

            frame_disp = None

            if vis:
                frame_disp = np.copy(frames[frame_id])

            """draw grid cells"""
            for grid_id in range(n_grid_cells):
                prev_grid_idy, prev_grid_idx = np.unravel_index(grid_id, grid_res)

                excel_idy, excel_idx = grid_to_excel_ids(prev_grid_idy, prev_grid_idx, grid_res)

                offset_cx, offset_cy = grid_cx[prev_grid_idy, prev_grid_idx], grid_cy[prev_grid_idy, prev_grid_idx]

                grid_box = np.array(
                    [offset_cx - grid_cell_size[0] / 2, offset_cy - grid_cell_size[1] / 2, grid_cell_size[0],
                     grid_cell_size[1]])

                if vis:
                    draw_box(frame_disp, grid_box, color='white', thickness=1)

            """show active objects and associated grid cells 
            """
            for _id, obj_id in enumerate(ann_idx):
                # traj_idx = annotations.traj_idx[obj_id]
                # curr_ann_data = annotations.data[, :]
                # curr_frame_ann_idx = np.flatnonzero(curr_ann_data[:, 0] == frame_id)
                # ann_idx = traj_idx[curr_frame_ann_idx]

                obj_data = annotations.data[obj_id, :]

                target_id = int(obj_data[1])
                traj_id = annotations.obj_to_traj[target_id]
                traj_vocab = vocab_annotations[traj_id]

                # obj_ann_idx = annotations.traj_idx[obj_id]
                active_grid_id = obj_grid_ids_flat[_id]

                grid_idy, grid_idx = np.unravel_index(active_grid_id, grid_res)

                cx, cy = grid_cx[grid_idy, grid_idx], grid_cy[grid_idy, grid_idx]

                grid_box = np.array([cx - grid_cell_size[0] / 2, cy - grid_cell_size[1] / 2,
                                     grid_cell_size[0], grid_cell_size[1]])
                obj_col = obj_cols[_id % n_obj_cols]

                # show('frame_disp', frame_disp, _pause=0)

                # traj_vocab['grid_cells'].append((grid_idy, grid_idx))

                if vocab_fmt == 0:
                    excel_idy, excel_idx = grid_to_excel_ids(grid_idy, grid_idx, grid_res)
                    word = excel_idx + excel_idy
                    if traj_vocab['sentence']:
                        traj_vocab['sentence'] += ' ' + word
                    else:
                        traj_vocab['sentence'] = word

                else:
                    if traj_vocab['sentence']:
                        _prev_grid_idy, _prev_grid_idx = prev_grid_ids[traj_id]
                        word = grid_to_direction(
                            (_prev_grid_idy, _prev_grid_idx),
                            (grid_idy, grid_idx),
                            max_diff=max_diff
                        )

                        traj_vocab['sentence'] += ' ' + word
                    else:
                        if vocab_fmt == 1:
                            word = f'R{grid_idy} C{grid_idx}'
                        elif vocab_fmt == 2:
                            word = f'R{grid_idy}C{grid_idx}'

                        traj_vocab['sentence'] = word

                vocab += word.split(' ')

                prev_grid_ids[traj_id] = (grid_idy, grid_idx)

                if vis:
                    # _id = f'{target_id}-{traj_id}-{win_id}'
                    _id = word
                    draw_box(frame_disp, obj_data[2:6], _id=_id, color='black', thickness=1, text_col='black')

                    draw_box(frame_disp, grid_box, color='red',
                             # transparency=0.1,
                             # _id=word, text_col='black',
                             thickness=1)

            if vis:
                frame_disp = resize_ar(frame_disp, height=960)

                if vis == 2:
                    out_img_id += 1
                    out_fname = 'image{:06d}.jpg'.format(out_img_id)
                    out_path = linux_path(out_dir, out_fname)
                    cv2.imwrite(out_path, frame_disp)

                _pause = show('frame_disp', frame_disp, _pause=_pause)

    vocab = sorted(list(set(vocab)))

    return vocab_annotations, traj_lengths, vocab


def diff_sentence_to_grid_cells(words, fmt_type, max_diff=1):
    n_dig = len(str(max_diff))
    unit_diff = max_diff == 1

    if fmt_type == 1:
        assert words[0][0] == 'R', f"Invalid first word {words[0]}"
        assert words[1][0] == 'C', f"Invalid second word {words[1]}"

        start_row_id = words[0][1:]
        start_col_id = words[1][1:]

        diff_words = words[2:]

    elif fmt_type == 2:

        assert words[0][0] == 'R' and words[0][n_dig] == 'C', f"Invalid first word {words[0]}"

        start_row_id = words[0][1:n_dig + 1]
        start_col_id = words[0][-n_dig:]

        diff_words = words[1:]

    else:
        raise AssertionError(f'invalid format type: {fmt_type}')

    grid_idy = int(start_row_id)
    grid_idx = int(start_col_id)

    grid_ids = [(grid_idy, grid_idx), ]

    prev_grid_idy, prev_grid_idx = grid_idy, grid_idx

    for word in diff_words:

        if word == 'I':
            grid_idy, grid_idx = prev_grid_idy, prev_grid_idx
        elif unit_diff:
            if word == 'S':
                grid_idy, grid_idx = prev_grid_idy + 1, prev_grid_idx
            elif word == 'N':
                grid_idy, grid_idx = prev_grid_idy - 1, prev_grid_idx
            elif word == 'E':
                grid_idy, grid_idx = prev_grid_idy, prev_grid_idx + 1
            elif word == 'W':
                grid_idy, grid_idx = prev_grid_idy, prev_grid_idx - 1

            elif word == 'SE':
                grid_idy, grid_idx = prev_grid_idy + 1, prev_grid_idx + 1

            elif word == 'SW':
                grid_idy, grid_idx = prev_grid_idy + 1, prev_grid_idx - 1

            elif word == 'NE':
                grid_idy, grid_idx = prev_grid_idy - 1, prev_grid_idx + 1

            elif word == 'NW':
                grid_idy, grid_idx = prev_grid_idy - 1, prev_grid_idx - 1
            else:
                raise AssertionError(f'invalid word: {word}')
        elif len(word) == n_dig + 1:
            direction = word[0]
            diff = int(word[1:])
            assert diff > 0, f"invalid diff: {diff} in word: {word}"

            if direction == 'S':
                grid_idy, grid_idx = prev_grid_idy + diff, prev_grid_idx
            elif direction == 'N':
                grid_idy, grid_idx = prev_grid_idy - diff, prev_grid_idx
            elif direction == 'E':
                grid_idy, grid_idx = prev_grid_idy, prev_grid_idx + diff
            elif direction == 'W':
                grid_idy, grid_idx = prev_grid_idy, prev_grid_idx - diff
            else:
                raise AssertionError(f'invalid word: {word}')
        elif len(word) == 2 * (n_dig + 1):
            direction = word[0] + word[n_dig+1]
            diff_y_, diff_x_= word[1:n_dig+1], word[n_dig + 2:]
            diff_y, diff_x = int(diff_y_), int(diff_x_)

            assert diff_y > 0, f"invalid diff_y: {diff_y} in word: {word}"
            assert diff_x > 0, f"invalid diff_x: {diff_x} in word: {word}"

            if direction == 'SE':
                grid_idy, grid_idx = prev_grid_idy + diff_y, prev_grid_idx + diff_x

            elif direction == 'SW':
                grid_idy, grid_idx = prev_grid_idy + diff_y, prev_grid_idx - diff_x

            elif direction == 'NE':
                grid_idy, grid_idx = prev_grid_idy - diff_y, prev_grid_idx + diff_x

            elif direction == 'NW':
                grid_idy, grid_idx = prev_grid_idy - diff_y, prev_grid_idx - diff_x
            else:
                raise AssertionError(f'invalid word: {word}')
        else:
            raise AssertionError(f'invalid word: {word}')

        grid_ids.append((grid_idy, grid_idx))

        prev_grid_idy, prev_grid_idx = grid_idy, grid_idx

    return grid_ids


def grid_to_direction(prev_grid_ids, curr_grid_ids, max_diff=1):
    prev_grid_idy, prev_grid_idx = prev_grid_ids
    grid_idy, grid_idx = curr_grid_ids

    unit_diff = max_diff == 1

    diff_y, diff_x = abs(grid_idy - prev_grid_idy), abs(grid_idx - prev_grid_idx)

    assert diff_y <= max_diff, \
        f'grid_idy {grid_idy} exceeds diff {max_diff} from prev_grid_idy: {prev_grid_idy}'

    assert diff_x <= max_diff, \
        f'grid_idx {grid_idx} exceeds diff {max_diff} from prev_grid_idx: {prev_grid_idx}'

    n_dig = len(str(max_diff))

    fmt = f'%0{n_dig}d'

    fmt_diff_y = fmt % diff_y
    fmt_diff_x = fmt % diff_x

    if grid_idy == prev_grid_idy:
        if grid_idx == prev_grid_idx:
            return 'I'

        if grid_idx > prev_grid_idx:
            return 'E' if unit_diff else f'E{fmt_diff_x}'

        if grid_idx < prev_grid_idx:
            return 'W' if unit_diff else f'W{fmt_diff_x}'

    if grid_idy > prev_grid_idy:
        if grid_idx == prev_grid_idx:
            return 'S' if unit_diff else f'S{fmt_diff_y}'

        if grid_idx > prev_grid_idx:
            return 'SE' if unit_diff else f'S{fmt_diff_x}E{fmt_diff_x}'

        if grid_idx < prev_grid_idx:
            return 'SW' if unit_diff else f'S{fmt_diff_y}W{fmt_diff_x}'

    if grid_idy < prev_grid_idy:
        if grid_idx == prev_grid_idx:
            return 'N' if unit_diff else f'N{fmt_diff_y}'

        if grid_idx > prev_grid_idx:
            return 'NE' if unit_diff else f'N{fmt_diff_y}E{fmt_diff_x}'

        if grid_idx < prev_grid_idx:
            return 'NW' if unit_diff else f'N{fmt_diff_y}W{fmt_diff_x}'


def excel_ids_to_grid(grid_res):
    excel_id_dict = {}
    grid_res_x, grid_res_y = grid_res

    for grid_idy in range(grid_res_y):
        for grid_idx in range(grid_res_x):
            excel_idy, excel_idx = grid_to_excel_ids(grid_idy, grid_idx)

            excel_id = f'{excel_idx}{excel_idy}'.lower()

            excel_id_dict[excel_id] = (grid_idy, grid_idx)

    return excel_id_dict


def grid_to_excel_ids(grid_idy, grid_idx, grid_res):

    n_dig_y, n_dig_x = len(str(grid_res[0])), len(str(grid_res[1]))

    fmt_y, fmt_x = f'%0{n_dig_y}d', f'%0{n_dig_x}d'

    excel_idy = fmt_y % grid_idy
    excel_idx = fmt_x % grid_idx

    # excel_idy = str(grid_idy + 1)
    # excel_idx_num_1 = int(grid_idx / 26)
    # if excel_idx_num_1 == 0:
    #     excel_idx = str(chr(grid_idx + 65))
    # else:
    #     excel_idx_num_2 = int(grid_idx % 26)
    #
    #     excel_idx_1 = str(chr(excel_idx_num_1 - 1 + 65))
    #     excel_idx_2 = str(chr(excel_idx_num_2 + 65))
    #
    #     excel_idx = excel_idx_1 + excel_idx_2

    return excel_idy, excel_idx


def build_targets_3d(frames, annotations, grid_res=(19, 19), frame_gap=1, win_size=100,
                     diff_grid_size=10, one_hot=1):
    """

    :param Annotations annotations:
    :param list[np.ndarray] frames:
    :param tuple(int, int) grid_res:
    :param int frame_gap: frame_gap
    :param int win_size: temporal window size
    :return:

    """
    n_frames = len(frames)
    frame_size = (frames[1].shape[1], frames[1].shape[0])

    diff_grid_res = np.array([frame_size[i] / diff_grid_size for i in range(2)])

    end_frame = n_frames - win_size
    grid_cell_size = np.array([frame_size[i] / grid_res[i] for i in range(2)])

    """grid cell centers
    """
    grid_x, grid_y = [np.arange(grid_cell_size[i] / 2.0, frame_size[i], grid_cell_size[i]) for i in range(2)]
    grid_cx, grid_cy = np.meshgrid(grid_x, grid_y)

    n_grid_cells = grid_cx.size

    ann_sizes = annotations.data[:, 4:6]
    ann_centers = annotations.data[:, 2:4] + annotations.data[:, 4:6] / 2.0

    for frame_id in range(0, end_frame, frame_gap):

        ann_idx = annotations.idx[frame_id]
        obj_centers = ann_centers[ann_idx, :]

        """Map each object to the grid that contains its centre
        """
        obj_grid_ids = (obj_centers.T / grid_cell_size[:, None]).astype(np.int64)
        # arr = np.array([[3, 6, 6], [4, 5, 1]])

        """np.ravel_multi_index takes row, col indices"""
        obj_grid_ids_flat = np.ravel_multi_index(obj_grid_ids[::-1], grid_res)

        u, c = np.unique(obj_grid_ids_flat, return_counts=True)
        dup_grid_ids = u[c > 1]
        active_grid_ids = list(u[c == 1])
        active_obj_ids = [np.nonzero(obj_grid_ids_flat == _id)[0].item() for _id in active_grid_ids]

        obj_cols = (
            'forest_green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
            'dark_slate_gray',
            'navy', 'turquoise')

        """Resolve multiple objects mapping to the same grid by choosing the nearest object in each case
        and add it to the list of active objects for this temporal window
        """
        for _id in dup_grid_ids:
            grid_ids = np.nonzero(obj_grid_ids_flat == _id)[0]

            dup_obj_locations = obj_centers[grid_ids, :]

            _id_2d = np.unravel_index(_id, grid_res)

            grid_center = np.array((grid_x[_id_2d[0]], grid_y[_id_2d[1]])).reshape((1, 2))

            dup_obj_distances = np.linalg.norm(dup_obj_locations - grid_center, axis=1)

            nearest_obj_id = grid_ids[np.argmin(dup_obj_distances)]

            active_obj_ids.append(nearest_obj_id)

            active_grid_ids.append(_id)

        active_obj_ids_ann = [int(annotations.data[ann_idx[i], 1]) for i in active_obj_ids]

        end_frame_id = min(frame_id + win_size, n_frames) - 1

        frame_disp = np.copy(frames[frame_id])

        """show grid cells"""
        for grid_id in range(n_grid_cells):
            prev_grid_idy, prev_grid_idx = np.unravel_index(grid_id, grid_res)

            offset_cx, offset_cy = grid_cx[prev_grid_idy, prev_grid_idx], grid_cy[prev_grid_idy, prev_grid_idx]

            grid_box = np.array(
                [offset_cx - grid_cell_size[0] / 2, offset_cy - grid_cell_size[1] / 2, grid_cell_size[0],
                 grid_cell_size[1]])

            draw_box(frame_disp, grid_box, _id=grid_id, color='black')

        """ahow active objects and associated grid cells 
        """
        for _id, obj_id in enumerate(active_obj_ids):
            # traj_idx = annotations.traj_idx[obj_id]
            # curr_ann_data = annotations.data[, :]
            # curr_frame_ann_idx = np.flatnonzero(curr_ann_data[:, 0] == frame_id)
            # ann_idx = traj_idx[curr_frame_ann_idx]

            obj_data = annotations.data[ann_idx[obj_id], :]

            # obj_ann_idx = annotations.traj_idx[obj_id]
            active_grid_id = active_grid_ids[_id]

            grid_idy, grid_idx = np.unravel_index(active_grid_id, grid_res)

            cx, cy = grid_cx[grid_idy, grid_idx], grid_cy[grid_idy, grid_idx]

            grid_box = np.array([cx - grid_cell_size[0] / 2, cy - grid_cell_size[1] / 2,
                                 grid_cell_size[0], grid_cell_size[1]])

            obj_col = obj_cols[_id % len(obj_cols)]

            draw_box(frame_disp, obj_data[2:6], _id=obj_data[1], color=obj_col)
            # show('frame_disp', frame_disp, _pause=0)

            draw_box(frame_disp, grid_box, color=obj_col)

        show('frame_disp', frame_disp, _pause=0)

        """Maximum possible distance between the centre of an object and the centres of 
        all of the neighbouring grid cells
        """
        max_dist = 1.5 * np.sqrt((grid_cell_size[0] ** 2 + grid_cell_size[1] ** 2))

        _pause = 100

        """compute distances and dist_probabilities for each active object wrt each of the 9 neighboring cells 
        in each frame of the current temporal window
        """
        """iterate over active objects from first frame of temporal window"""
        for _id, obj_id in enumerate(active_obj_ids_ann):

            # obj_id2 = active_obj_ids[_id]

            """all annotations for this object in the temporal window"""
            obj_ann_idx = annotations.traj_idx[obj_id]
            obj_ann_idx = [k for k in obj_ann_idx if annotations.data[k, 0] <= end_frame_id]
            obj_ann_data = annotations.data[obj_ann_idx, :]

            curr_obj_sizes = ann_sizes[obj_ann_idx, :]
            curr_obj_centers = ann_centers[obj_ann_idx, :]
            """Map each object to the grid that contains its centre
            """
            curr_obj_grid_ids = (curr_obj_centers.T / grid_cell_size[:, None]).astype(np.int64)
            # arr = np.array([[3, 6, 6], [4, 5, 1]])

            """np.ravel_multi_index takes row, col indices"""

            curr_obj_grid_ids_flat = np.ravel_multi_index(curr_obj_grid_ids[::-1], grid_res)

            obj_col = obj_cols[_id % len(obj_cols)]

            _prev_grid_id = None
            _obj_centers_rec = []
            _obj_centers = []
            _one_hot_obj_centers_rec = []

            """Iterate over objects corresponding to this target in each frame of the temporal window"""
            for temporal_id, curr_grid_id in enumerate(curr_obj_grid_ids_flat):
                if _prev_grid_id is None:
                    _prev_grid_id = curr_grid_id

                prev_grid_idy, prev_grid_idx = np.unravel_index(_prev_grid_id, grid_res)
                curr_grid_idy, curr_grid_idx = np.unravel_index(curr_grid_id, grid_res)

                obj_cx, obj_cy = curr_obj_centers[temporal_id, :]
                obj_w, obj_h = curr_obj_sizes[temporal_id, :]

                prev_cx, prev_cy = grid_cx[prev_grid_idy, prev_grid_idx], grid_cy[prev_grid_idy, prev_grid_idx]
                curr_cx, curr_cy = grid_cx[curr_grid_idy, curr_grid_idx], grid_cy[curr_grid_idy, curr_grid_idx]

                prev_grid_box = np.array([prev_cx - grid_cell_size[0] / 2, prev_cy - grid_cell_size[1] / 2,
                                          grid_cell_size[0], grid_cell_size[1]])
                curr_grid_box = np.array([curr_cx - grid_cell_size[0] / 2, curr_cy - grid_cell_size[1] / 2,
                                          grid_cell_size[0], grid_cell_size[1]])

                diff_cx, diff_cy = curr_cx - prev_cx, curr_cy - prev_cy

                """find quadrant of motion direction
                """
                if diff_cx >= 0:
                    """move right - quadrant 1 or 4"""
                    if diff_cy >= 0:
                        """move up"""
                        quadrant = 1
                    else:
                        """move down"""
                        quadrant = 4
                else:
                    """move left - quadrant 2 or 3"""
                    if diff_cy >= 0:
                        """move up"""
                        quadrant = 2
                    else:
                        """move down"""
                        quadrant = 3

                # dist_probabilities = np.zeros((3, 3))
                distances = []
                distances_inv = []
                dist_ids = []
                grid_centers = []

                neigh_grid_ids = []

                one_hot_probabilities = np.zeros((9,), dtype=np.float32)
                if prev_grid_idy == curr_grid_idy:
                    """middle row"""
                    if prev_grid_idx == curr_grid_idx:
                        class_id = 4
                    elif prev_grid_idx > curr_grid_idx:
                        class_id = 3
                    else:
                        class_id = 5
                elif prev_grid_idy > curr_grid_idy:
                    """bottom row"""
                    if prev_grid_idx == curr_grid_idx:
                        class_id = 7
                    elif prev_grid_idx > curr_grid_idx:
                        class_id = 6
                    else:
                        class_id = 8
                else:
                    """top row"""
                    if prev_grid_idx == curr_grid_idx:
                        class_id = 4
                    elif prev_grid_idx > curr_grid_idx:
                        class_id = 1
                    else:
                        class_id = 7

                one_hot_probabilities[class_id] = 1
                obj_cx_rec, obj_cy_rec = grid_cx[curr_grid_idy, curr_grid_idx], grid_cy[curr_grid_idy, curr_grid_idx]

                _one_hot_obj_centers_rec.append((obj_cx_rec, obj_cy_rec))
                _obj_centers.append((obj_cx, obj_cy))

                """smooth probabilities based on object's distance from grid centers"""

                """Iterate over neighbouring grid cells"""
                for prob_idx, offset_x in enumerate((-1, 0, 1)):
                    offset_idx = prev_grid_idx + offset_x
                    if offset_idx >= grid_res[0]:
                        continue
                    for prob_idy, offset_y in enumerate((-1, 0, 1)):
                        offset_idy = prev_grid_idy + offset_y
                        if offset_idy >= grid_res[1]:
                            continue

                        neigh_grid_ids.append((offset_idy, offset_idx))

                        offset_cx, offset_cy = grid_cx[offset_idy, offset_idx], grid_cy[offset_idy, offset_idx]
                        dist = np.sqrt((obj_cx - offset_cx) ** 2 + (obj_cy - offset_cy) ** 2) / max_dist

                        assert 0 <= dist <= max_dist, f"Invalid distance: {dist}"

                        norm_dist = dist / max_dist

                        """Large distance = small probability"""
                        # inv_dist = 1.0 / (1.0 + dist)
                        inv_dist = 1.0 - norm_dist

                        distances.append(norm_dist)
                        distances_inv.append(inv_dist)
                        dist_ids.append((prob_idy, prob_idx))
                        grid_centers.append((offset_cx, offset_cy))

                distances = np.asarray(distances)
                distances_inv = np.asarray(distances_inv)
                grid_centers = np.asarray(grid_centers)

                """sum to 1"""
                # dist_probabilities = np.exp(distances_inv) / sum(np.exp(distances_inv))
                distances_inv_sum = np.sum(distances_inv)
                dist_inv_probabilities = distances_inv / distances_inv_sum

                distances_sum = np.sum(distances)
                dist_probabilities = distances / distances_sum
                dist_probabilities2 = 1 - dist_probabilities

                eps = np.finfo(np.float32).eps
                dist_probabilities3 = 1.0 / (eps + dist_probabilities)
                dist_probabilities3_sum = np.sum(dist_probabilities3)
                dist_probabilities4 = dist_probabilities3 / dist_probabilities3_sum

                obj_cx_rec = np.average(grid_centers[:, 0], weights=dist_probabilities4)
                obj_cy_rec = np.average(grid_centers[:, 1], weights=dist_probabilities4)

                _obj_centers_rec.append((obj_cx_rec, obj_cy_rec))

                obj_cx_diff = obj_cx_rec - obj_cx
                obj_cy_diff = obj_cy_rec - obj_cy

                obj_box = np.array([obj_cx - obj_w / 2, obj_cy - obj_h / 2,
                                    obj_cx + obj_w / 2, obj_cy + obj_h / 2])

                obj_box_rec = np.array([obj_cx_rec - obj_w / 2, obj_cy_rec - obj_h / 2,
                                        obj_cx_rec + obj_w / 2, obj_cy_rec + obj_h / 2])

                obj_box_rec_iou = np.empty((1,))
                compute_overlap(obj_box_rec_iou, None, None, obj_box.reshape((1, 4)),
                                obj_box_rec.reshape((1, 4)))

                min_probability, max_probability = np.min(dist_probabilities4), np.max(dist_probabilities4)
                dist_probabilities_norm = (dist_probabilities4 - min_probability) / (max_probability - min_probability)

                """Large distance = small probability"""
                # dist_probabilities = 1.0 - dist_probabilities

                """show dist_probabilities as a 2D image with 3x3 grid
                """
                prob_img = np.zeros((300, 300, 3), dtype=np.uint8)
                one_hot_prob_img = np.zeros((300, 300, 3), dtype=np.uint8)

                for _prob_id, _prob in enumerate(dist_probabilities_norm):
                    prob_idy, prob_idx = dist_ids[_prob_id]
                    _prob_col = prob_to_rgb2(_prob)
                    r, g, b = _prob_col
                    start_row = prob_idy * 100
                    start_col = prob_idx * 100

                    end_row = start_row + 100
                    end_col = start_col + 100

                    prob_img[start_row:end_row, start_col:end_col, :] = (b, g, r)

                    one_hot_prob_col = prob_to_rgb2(one_hot_probabilities[_prob_id])
                    r, g, b = one_hot_prob_col
                    one_hot_prob_img[start_row:end_row, start_col:end_col, :] = (b, g, r)

                obj_data = obj_ann_data[temporal_id, :]

                curr_frame_id = int(obj_data[0])
                curr_frame = frames[curr_frame_id]

                curr_frame_disp_grid = np.copy(curr_frame)

                """draw all grid cells"""
                for grid_id in range(n_grid_cells):
                    _grid_idy, _grid_idx = np.unravel_index(grid_id, grid_res)

                    _cx, _cy = grid_cx[_grid_idy, _grid_idx], grid_cy[_grid_idy, _grid_idx]

                    grid_box = np.array(
                        [_cx - grid_cell_size[0] / 2, _cy - grid_cell_size[1] / 2, grid_cell_size[0],
                         grid_cell_size[1]])

                    # col = 'red' if (prev_grid_idy, prev_grid_idx) in neigh_grid_ids else 'black'
                    draw_box(curr_frame_disp_grid, grid_box, color='black')

                """draw neighboring grid cells"""
                col = 'green' if _prev_grid_id == curr_grid_id else 'red'
                for _grid_idy, _grid_idx in neigh_grid_ids:
                    _cx, _cy = grid_cx[_grid_idy, _grid_idx], grid_cy[_grid_idy, _grid_idx]

                    grid_box = np.array(
                        [_cx - grid_cell_size[0] / 2, _cy - grid_cell_size[1] / 2, grid_cell_size[0],
                         grid_cell_size[1]])
                    draw_box(curr_frame_disp_grid, grid_box, color=col)

                draw_box(curr_frame_disp_grid, obj_data[2:6], _id=obj_data[1], color='blue')

                curr_frame_traj_rec = np.copy(curr_frame)
                draw_box(curr_frame_traj_rec, obj_data[2:6], color='blue', thickness=1)
                draw_traj2(curr_frame_traj_rec, _obj_centers_rec, color='red')
                draw_traj2(curr_frame_traj_rec, _obj_centers, color='green')
                # curr_frame_traj_rec = resize_ar(curr_frame_traj_rec, 1920, 1080)

                curr_frame_traj_one_hot_rec = np.copy(curr_frame)
                draw_box(curr_frame_traj_one_hot_rec, obj_data[2:6], color='blue', thickness=1)
                draw_traj2(curr_frame_traj_one_hot_rec, _one_hot_obj_centers_rec, color='red')
                draw_traj2(curr_frame_traj_one_hot_rec, _obj_centers, color='green')
                # curr_frame_traj_one_hot_rec = resize_ar(curr_frame_traj_one_hot_rec, 1920, 1080)

                # curr_frame_traj = np.copy(curr_frame)
                # draw_box(curr_frame_traj, obj_data[2:6], color='blue', thickness=1)
                # draw_traj2(curr_frame_traj, _obj_centers, color='green')
                # curr_frame_traj = resize_ar(curr_frame_traj, 1920, 1080)

                for _obj_cx_rec, _obj_cy_rec in _obj_centers_rec:
                    cv2.circle(curr_frame_disp_grid, (int(_obj_cx_rec), int(_obj_cy_rec)), 1, color=(255, 255, 255),
                               thickness=2)

                curr_frame_disp = np.copy(curr_frame)

                draw_box(curr_frame_disp, prev_grid_box, color='red')
                draw_box(curr_frame_disp, curr_grid_box, color='green')

                draw_box(curr_frame_disp, obj_data[2:6], _id=obj_data[1], color=obj_col)

                show('curr_frame_disp_grid', curr_frame_disp_grid, _pause=0)
                show('curr_frame_disp', curr_frame_disp, _pause=0)
                show('one_hot_prob_img', one_hot_prob_img, _pause=0)
                # show('curr_frame_traj', curr_frame_traj, _pause=0)
                show('curr_frame_traj_rec', curr_frame_traj_rec, _pause=0)
                show('curr_frame_traj_one_hot_rec', curr_frame_traj_one_hot_rec, _pause=0)
                _pause = show('prob_img', prob_img, _pause=_pause)

                _prev_grid_id = curr_grid_id

                # print()

            print()

        print()


def build_targets_seq(frames, annotations, frame_gap=1, win_size=50):
    """

    :param Annotations annotations:
    :param list[np.ndarray] frames:
    :param tuple(int, int) grid_res:
    :param int frame_gap: frame_gap
    :param int win_size: temporal window size
    :return:

    """
    n_frames = len(frames)
    frame_size = (frames[1].shape[1], frames[1].shape[0])

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")

    out_dir = linux_path('log', 'build_targets_seq_{}'.format(timestamp))
    os.makedirs(out_dir, exist_ok=1)

    end_frame = n_frames - win_size

    ann_mins = annotations.data[:, 2:4]
    ann_maxs = annotations.data[:, 2:4] + annotations.data[:, 4:6]
    ann_sizes = annotations.data[:, 4:6]
    ann_obj_ids = annotations.data[:, 1]
    ann_centers = annotations.data[:, 2:4] + annotations.data[:, 4:6] / 2.0

    out_img_id = 0

    temporal_win_id = 0

    for frame_id in range(0, end_frame, frame_gap):

        ann_idx = annotations.idx[frame_id]
        obj_centers = ann_centers[ann_idx, :]
        obj_mins = ann_mins[ann_idx, :]
        obj_maxs = ann_maxs[ann_idx, :]

        curr_frame_data = annotations.data[ann_idx, :]

        curr_ann_obj_ids = ann_obj_ids[ann_idx]

        # obj_cols = (
        #     'forest_green', 'blue', 'red', 'cyan', 'magenta', 'gold', 'purple', 'peach_puff', 'azure',
        #     'dark_slate_gray',
        #     'navy', 'turquoise')

        end_frame_id = min(frame_id + win_size, n_frames) - 1

        _pause = 100

        temporal_win_id += 1

        label_txt = 'temporal window {} :: frames {} --> {}'.format(temporal_win_id, frame_id + 1, end_frame_id + 1)

        """compute distances and dist_probabilities for each active object wrt each of the 9 neighboring cells 
        in each frame of the current temporal window
        """
        """iterate over active objects from first frame of temporal window"""
        input_attention_map = np.full(frames[1].shape[:2], 255, dtype=np.uint8)

        for _id, obj_id in enumerate(curr_ann_obj_ids):
            traj_id = annotations.obj_to_traj[obj_id]
            # obj_id2 = active_obj_ids[_id]

            """all annotations for this object in the temporal window"""
            obj_ann_idx = annotations.traj_idx[traj_id]
            obj_ann_idx = [k for k in obj_ann_idx if annotations.data[k, 0] <= end_frame_id]
            obj_ann_data = annotations.data[obj_ann_idx, :]

            curr_obj_sizes = ann_sizes[obj_ann_idx, :]
            curr_obj_centers = ann_centers[obj_ann_idx, :]
            curr_obj_mins = ann_mins[obj_ann_idx, :]
            curr_obj_maxs = ann_maxs[obj_ann_idx, :]

            # obj_col = obj_cols[_id % len(obj_cols)]

            _prev_grid_id = None
            _obj_centers_rec = []
            _obj_boxes = []
            _obj_centers = []
            _one_hot_obj_centers_rec = []

            n_obj_ann_data = obj_ann_data.shape[0]

            all_objects = np.copy(frames[frame_id])

            for __id, __obj_id in enumerate(curr_ann_obj_ids):
                if __id == _id:
                    col = 'green'
                elif __id < _id:
                    col = 'red'
                else:
                    col = 'black'

                draw_box(all_objects, curr_frame_data[__id, 2:6], color=col, thickness=2)

            obj_minx, obj_miny = curr_obj_mins[0, :].astype(np.int)
            obj_maxx, obj_maxy = curr_obj_maxs[0, :].astype(np.int)

            output_attention_map = np.full(frames[1].shape[:2], 0, dtype=np.uint8)
            output_attention_map[obj_miny:obj_maxy, obj_minx:obj_maxx] = 255
            _label_txt = label_txt + ' object {}'.format(_id + 1)

            # show('input_attention_map', input_attention_map,
            #      )
            # show('output_attention_map', output_attention_map,
            #      )
            """Iterate over objects corresponding to this target in each frame of the temporal window"""
            for temporal_id, curr_grid_id in enumerate(range(n_obj_ann_data)):
                obj_cx, obj_cy = curr_obj_centers[temporal_id, :]
                obj_data = obj_ann_data[temporal_id, :]

                # obj_w, obj_h = curr_obj_sizes[temporal_id, :]
                # obj_minx, obj_miny = curr_obj_mins[temporal_id, :]
                # obj_maxx, obj_maxy = curr_obj_maxs[temporal_id, :]

                _obj_centers.append((obj_cx, obj_cy))

                curr_frame_id = int(obj_data[0])
                curr_frame = frames[curr_frame_id]

                _obj_boxes.append(obj_data[2:6])

                output_boxes = np.copy(curr_frame)
                # draw_box(output_boxes, obj_data[2:6], color='green', thickness=1)
                draw_traj2(output_boxes, _obj_centers, _obj_boxes, color='green')
                # output_boxes = resize_ar(output_boxes, 1600, 900)

                # annotate_and_show('output_boxes', output_boxes, text=_label_txt, n_modules=0)

                all_images = [input_attention_map, output_attention_map, all_objects, output_boxes]
                img_labels = ['input attention_map',
                              'output attention_map ',
                              'all objects',
                              'output boxes']

                sequence_prediction_mot = annotate_and_show('sequence_prediction_mot', all_images, text=_label_txt,
                                                            n_modules=0, grid_size=(2, 2), img_labels=img_labels,
                                                            only_annotate=1)

                cv2.imshow('sequence_prediction_mot', sequence_prediction_mot)
                cv2.waitKey(1)

                out_img_id += 1
                out_fname = 'image{:06d}.jpg'.format(out_img_id)

                out_path = linux_path(out_dir, out_fname)
                cv2.imwrite(out_path, sequence_prediction_mot)

            # annotate_and_show('all_objects', all_objects, text=_label_txt, n_modules=0)

            input_attention_map[obj_miny:obj_maxy, obj_minx:obj_maxx] = 0
