import os

# script_dir = os.path.dirname(os.path.abspath(__file__))
# script_parent_dir = script_dir.replace(os.sep, '/') + '/..'


import sys
import math

# sys.path.append(script_parent_dir)
sys.path.append('../isl_labeling_tool/deep_mdp')

import pandas as pd

import numpy as np
# import random
# import time
import json

from tqdm import tqdm
from datetime import datetime

import functools

import cv2
import copy
import paramparse

"""deep mdp modules"""
from input import Input
from objects import Annotations
from data import Data

from utilities import CustomLogger, SIIF, linux_path, draw_box, resize_ar, show, annotate_and_show
from utilities import CVText, col_bgr

from dnc_utilities import excel_ids_to_grid, diff_sentence_to_grid_cells


class Params:
    """
    :ivar mode:
    0: build_targets_densecap
    1: build_targets_seq

    """

    class SlidingWindow:
        num = 0
        sample = 0
        size = 0
        stride = 0

    def __init__(self):
        self.gpu = ''
        self.cfg = ('',)

        self.json = ''

        self.set = ''
        self.seq = ()

        self.mode = 0
        self.load = 0
        self.save = 1
        self.start = 0

        self.grid_res = (32, 32)
        self.frame_gap = 1
        self.fps = 30
        self.vis = 1

        self.win_size = 0

        self.vocab_fmt = 0
        self.max_diff = 1

        self.n_proc = 1

        self.slide = Params.SlidingWindow()
        self.input = Input.Params(source_type=-1, batch_mode=False)
        self.data = Data.Params()
        self.ann = Annotations.Params()


def draw_grid_cell(frame_disp, grid_cell, grid_centers, grid_cell_size, color='white', thickness=1):
    grid_idy, grid_idx = grid_cell
    grid_cy, grid_cx = grid_centers

    offset_cx, offset_cy = grid_cx[grid_idy, grid_idx], grid_cy[grid_idy, grid_idx]

    grid_box = np.array(
        [offset_cx - grid_cell_size[0] / 2, offset_cy - grid_cell_size[1] / 2, grid_cell_size[0],
         grid_cell_size[1]])

    draw_box(frame_disp, grid_box, color=color, thickness=thickness)


def compress_traj(grid_ids, start_frame, end_frame):
    n_frames = end_frame - start_frame + 1
    n_grid_ids = len(grid_ids)

    assert n_grid_ids > n_frames, "trajectory size must exceed n_frames for compression"

    print(f'compressing trajectory from {n_grid_ids} to {n_frames}')

    frame_to_traj_dict = {
        start_frame: grid_ids[0],
        end_frame: grid_ids[-1],
    }
    skip_ratio = float(n_grid_ids - 2) / float(n_frames - 2)
    assigned_indices = [0, n_grid_ids - 1]
    for frame_id in range(start_frame + 1, end_frame):
        grid_ids_index = int(math.floor(skip_ratio * (frame_id - start_frame)))
        assert grid_ids_index not in assigned_indices, f"already assigned grid_ids_index: {grid_ids_index}"
        assigned_indices.append(grid_ids_index)
        frame_to_traj_dict[frame_id] = grid_ids[grid_ids_index]

    return frame_to_traj_dict


def expand_traj(grid_cells, start_frame, end_frame, frames, disp_fn):
    n_frames = end_frame - start_frame + 1
    n_grid_ids = len(grid_cells)

    assert n_grid_ids >= 2, "there must be at least two grid cells to interpolate from"

    assert n_grid_ids < n_frames, "trajectory size must not exceed n_frames for expansion"

    print(f'expanding trajectory from {n_grid_ids} to {n_frames}')

    frame_to_traj_dict = {
        start_frame: grid_cells[0],
        end_frame: grid_cells[-1],
    }
    skip_ratio = float(n_frames) / float(n_grid_ids)
    assigned_frames = [start_frame, end_frame]

    for grid_cells_index in range(1, n_grid_ids - 1):
        frame_id = start_frame + int(math.floor(skip_ratio * grid_cells_index))

        assert frame_id not in assigned_frames, f"already assigned frame_id: {frame_id}"

        assigned_frames.append(frame_id)
        frame_to_traj_dict[frame_id] = grid_cells[grid_cells_index]

    unassigned_frames = [k for k in range(start_frame, end_frame + 1) if k not in assigned_frames]

    assert len(unassigned_frames) == n_frames - n_grid_ids, "something weird going on"

    _pause = 1

    out_grid_cells = list(grid_cells[:])

    for i, unassigned_frame in enumerate(unassigned_frames):
        future_assigned_frames = [assigned_frame for assigned_frame in assigned_frames if
                                  assigned_frame > unassigned_frame]
        past_assigned_frames = [assigned_frame for assigned_frame in assigned_frames if
                                assigned_frame < unassigned_frame]

        future_dists = [assigned_frame - unassigned_frame for assigned_frame in future_assigned_frames]
        past_dists = [unassigned_frame - assigned_frame for assigned_frame in past_assigned_frames]

        future_nn_frames = np.argsort(future_dists)
        past_nn_frames = np.argsort(past_dists)

        future_nn, past_nn = int(future_nn_frames[0]), int(past_nn_frames[0])

        future_frame, past_frame = future_assigned_frames[future_nn], past_assigned_frames[past_nn]

        future_dist, past_dist = future_dists[future_nn], past_dists[past_nn]

        assert future_dist > 0, "future_dist must be > 0"
        assert past_dist > 0, "past_dist must be > 0"

        future_idy, future_idx = frame_to_traj_dict[future_frame]
        past_idy, past_idx = frame_to_traj_dict[past_frame]

        total_dist = future_dist + past_dist

        assert total_dist > 0, "total_dist must be > 0"

        grid_idy = int(round(past_idy + (future_idy - past_idy) * past_dist / total_dist))
        grid_idx = int(round(past_idx + (future_idx - past_idx) * past_dist / total_dist))

        interp_grid_cell = (grid_idy, grid_idx)

        # frame_disp = np.copy(frames[unassigned_frame])
        # for grid_cell in grid_cells:
        #     disp_fn(frame_disp, grid_cell, color='green')
        #
        # text = f'{unassigned_frame} ({grid_idy}, {grid_idx})\n' \
        #     f'{past_frame} ({past_idy}, {past_idx})-->{past_dist}\n' \
        #     f'{future_frame}({future_idy}, {future_idx})-->{future_dist}'
        #
        # disp_fn(frame_disp, (future_idy, future_idx), color='white')
        # disp_fn(frame_disp, (past_idy, past_idx), color='black')
        # disp_fn(frame_disp, interp_grid_cell, color='gray')
        # _pause = annotate_and_show('expand_traj', frame_disp,
        #                            text=text,
        #                            pause=_pause, n_modules=0)

        frame_to_traj_dict[unassigned_frame] = interp_grid_cell

        out_grid_cells.append(interp_grid_cell)
        # assigned_frames.append(unassigned_frame)

    return out_grid_cells, frame_to_traj_dict


def run(seq_info, dnc_data, frames, json_data, sentence_to_grid_cells, n_seq, out_dir,
        grid_res, fps, vis,
        params: Params):
    if frames is None:
        seq_id, seq_suffix, start_id, end_id = seq_info

        _logger = CustomLogger.setup(__name__)

        _data = Data(params.data, _logger)

        if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
            _logger.error('Data module could not be initialized')
            return None

        subset = "training" if _data.split == 'train' else "validation"

        input_params = copy.deepcopy(params.input)  # type: Input.Params

        """end_id is exclusive but Input expects inclusive"""
        input_params.frame_ids = (start_id, end_id - 1)

        input_params.batch_mode = 1

        _input = Input(input_params, _logger)
        seq_name = _data.seq_name

        if seq_suffix:
            seq_name = f'{seq_name}--{seq_suffix}'

        print(f'\nseq {seq_id + 1} / {n_seq}: {seq_name}\n')

        if not _input.initialize(_data):
            _logger.error('Input pipeline could not be initialized')
            return False

        # read detections and annotations
        if not _input.read_annotations():
            _logger.error('Annotations could not be read')
            return False

        _n_frames = _input.n_frames

        # duration = float(_n_frames) / params.fps
        _annotations = _input.annotations  # type: Annotations
        # _detections = _input.detections  # type: Detections

        frames = _input.all_frames

        if dnc_data is None:
            dnc_data = json_data[seq_name]

            if isinstance(dnc_data, dict):
                dnc_data = dnc_data['annotations']

    frame_res = frames[0].shape[:2]

    grid_cell_size = np.array([frame_res[i] / grid_res[i] for i in range(2)])

    grid_x, grid_y = [np.arange(grid_cell_size[i] / 2.0, frame_res[i], grid_cell_size[i]) for i in range(2)]
    grid_cx, grid_cy = np.meshgrid(grid_x, grid_y)

    frame_disp_dict = {}

    for traj_id, traj_datum in enumerate(dnc_data):
        sentence = traj_datum["sentence"].upper()
        timestamp = traj_datum["segment"]

        start_t, end_t = timestamp

        start_frame, end_frame = int(start_t * params.fps), int(end_t * params.fps)
        traj_n_frames = end_frame - start_frame + 1

        words = sentence.split(' ')

        grid_cells = sentence_to_grid_cells(words)

        n_grid_cells = len(grid_cells)
        disp_fn = functools.partial(draw_grid_cell,
                                    grid_centers=(grid_cy, grid_cx),
                                    grid_cell_size=grid_cell_size)

        # frame = _input.all_frames[start_frame]
        if n_grid_cells > traj_n_frames:
            frame_to_grid_cell = compress_traj(grid_cells, start_frame, end_frame)
        elif n_grid_cells < traj_n_frames:
            grid_cells, frame_to_grid_cell = expand_traj(grid_cells, start_frame, end_frame, frames, disp_fn)
        else:
            frame_to_grid_cell = {
                frame_id: grid_cells[frame_id - start_frame]
                for frame_id in range(start_frame, end_frame + 1)
            }

        _pause = 1

        frame_disp_list = []

        for frame_id in range(start_frame, end_frame + 1):
            if params.vis:
                frame_disp = np.copy(frames[frame_id])

                for grid_cell in grid_cells:
                    disp_fn(frame_disp, grid_cell, color='green')

                grid_cell = frame_to_grid_cell[frame_id]

                disp_fn(frame_disp, grid_cell, color='white', thickness=2)

                header_fmt = CVText()
                location = (header_fmt.location + header_fmt.offset[0], header_fmt.location + header_fmt.offset[1])
                color = col_bgr[header_fmt.color]
                cv2.putText(frame_disp, f'frame {frame_id}', location, header_fmt.font,
                            header_fmt.size, color, header_fmt.thickness, header_fmt.line_type)

                frame_disp = resize_ar(frame_disp, height=960)

                if show:
                    _pause = show('frame_disp', frame_disp, _pause=_pause)
                else:
                    frame_disp_list.append(frame_disp)

        frame_disp_dict[traj_id] = frame_disp_list

    return frame_disp_list


def main():
    """
    converts annotations from MOT format into JSON and CSV for training densecap
    with optional temporal sliding windows that might be useful for inference
    """
    params = Params()
    paramparse.process(params)

    SIIF.setup()

    _logger = CustomLogger.setup(__name__)
    _data = Data(params.data, _logger)

    try:
        params.set = int(params.set)
    except ValueError:
        params.set = params.data.name_to_id(params.set)

    set_name = _data.sets[params.set]
    n_sequences = len(_data.sequences[set_name])

    seq_ids = params.seq

    if not seq_ids:
        seq_ids = tuple(range(n_sequences))

    interval = params.slide.sample
    if interval <= 0:
        interval = 1

    assert params.json, "json file must be provided"
    if os.path.isdir(params.json):
        params.json = linux_path(params.json, 'densecap.json')

    assert os.path.isfile(params.json), f"invalid json file: {params.json}"

    with open(params.json, 'r') as fid:
        json_data = json.load(fid)

    if 'database' in json_data:
        json_data = json_data['database']

    if params.vocab_fmt == 0:
        word_to_grid_cell = excel_ids_to_grid(params.grid_res)
        sentence_to_grid_cells = lambda words: [word_to_grid_cell[word] for word in words]
    else:
        sentence_to_grid_cells = functools.partial(diff_sentence_to_grid_cells,
                                                   fmt_type=params.vocab_fmt,
                                                   max_diff=params.max_diff,
                                                   )
    seq_info_list = []
    pbar = tqdm(seq_ids)

    enable_slide = params.slide.size or params.slide.stride or params.slide.sample
    for seq_id in pbar:

        if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
            _logger.error('Data module could not be initialized')
            return None

        start_id = 0
        seq_name = _data.seq_name
        seq_n_frames = _data.seq_n_frames

        assert seq_n_frames % interval == 0, f"interval {interval} does not divide seq_n_frames {seq_n_frames} evenly"

        win_size = params.slide.size
        if win_size <= 0:
            win_size = int(_data.seq_n_frames / interval)

        win_stride = params.slide.stride
        if win_stride <= 0:
            win_stride = win_size
        win_id = 0
        while True:
            abs_start_id = int(start_id * interval)

            if abs_start_id >= seq_n_frames or win_id >= params.slide.num > 0:
                break

            if abs_start_id >= seq_n_frames:
                break

            end_id = start_id + win_size

            abs_end_id = int(end_id * interval)

            if abs_end_id >= seq_n_frames:
                abs_end_id = seq_n_frames
                end_id = int(abs_end_id / interval)

            if enable_slide:
                suffix = f'{abs_start_id}_{abs_end_id}'
            else:
                suffix = ''

            seq_info_list.append((seq_id, suffix, abs_start_id, abs_end_id))

            print(f'{seq_name}: {abs_start_id} to {abs_end_id}')

            start_id += win_stride

            win_id += 1

    n_seq = len(seq_info_list)

    # exit()

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    out_dir = linux_path('log', f'mot_to_dnc', f'{set_name}_{timestamp}')
    os.makedirs(out_dir, exist_ok=1)

    traj_lengths_out_dir = linux_path(out_dir, 'traj_lengths')
    os.makedirs(traj_lengths_out_dir, exist_ok=1)

    database = {}
    duration_frame_csv_rows = []
    all_traj_lengths = []
    seq_to_traj_lengths = []

    n_trajectories = 0

    n_proc = min(params.n_proc, n_seq)

    func = functools.partial(
        run,
        dnc_data=None,
        frames=None,
        n_seq=n_seq,
        json_data=json_data,
        sentence_to_grid_cells=sentence_to_grid_cells,
        out_dir=out_dir,
        # traj_lengths_out_dir=traj_lengths_out_dir,
        params=params,
    )

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

    for seq_id, traj_lengths, duration_frame_csv_row, vocab_annotations, subset in tqdm(
            results, desc='postprocessing results'):
        seq_name = duration_frame_csv_row['name']
        duration = duration_frame_csv_row['duration']

        duration_frame_csv_rows.append(duration_frame_csv_row)

        seq_to_traj_lengths += [(seq_name, traj_length) for traj_length in traj_lengths]

        all_traj_lengths += traj_lengths

        n_trajectories += len(vocab_annotations)

        database[seq_name] = {
            'duration': duration,
            "subset": subset,
            "annotations": vocab_annotations,
        }

    mean_traj_length = np.mean(all_traj_lengths)
    std_traj_length = np.std(all_traj_lengths)
    median_traj_length = np.median(all_traj_lengths)
    min_traj_length = np.amin(all_traj_lengths)
    max_traj_length = np.amax(all_traj_lengths)

    print(f'\nall traj_length: '
          f'mean: {mean_traj_length} '
          f'median: {median_traj_length} '
          f'min: {min_traj_length} '
          f'max: {max_traj_length} '
          f'std: {std_traj_length} '
          )

    seq_to_traj_lengths_out_path = linux_path(traj_lengths_out_dir, f'seq_to_traj_lengths.txt')

    seq_to_traj_lengths_str = '\n'.join(
        f'{seq_name}\t{traj_length}' for seq_name, traj_length in seq_to_traj_lengths)

    with open(seq_to_traj_lengths_out_path, 'w') as fid:
        fid.write(seq_to_traj_lengths_str)

    traj_lengths_out_path = linux_path(traj_lengths_out_dir, f'all.txt')
    np.savetxt(traj_lengths_out_path, np.asarray(all_traj_lengths, dtype=np.uint32), fmt='%d')

    json_dict = dict(
        database=database
    )

    json_path = linux_path(out_dir, f'{set_name}_annotations_trainval.json')
    print(f'saving json with {n_seq} sequences and {n_trajectories} trajectories')
    print(f'json_path: {json_path}')
    with open(json_path, 'w') as f:
        output_json_data = json.dumps(json_dict, indent=4)
        f.write(output_json_data)

    csv_path = os.path.join(out_dir, f'{set_name}_duration_frame.csv')
    print(f'csv_path: {csv_path}')
    df = pd.DataFrame(duration_frame_csv_rows)
    df.to_csv(csv_path, index=False, header=False)

    print(f'traj_lengths_out_dir: {traj_lengths_out_dir}')
    print(f'out_dir: {out_dir}')


if __name__ == "__main__":
    main()
