import os

# script_dir = os.path.dirname(os.path.abspath(__file__))
# script_parent_dir = script_dir.replace(os.sep, '/') + '/..'


import sys

# sys.path.append(script_parent_dir)
sys.path.append('../isl_labeling_tool/deep_mdp')

import pandas as pd

import numpy as np
# import random
# import time
import json

from tqdm import tqdm
from datetime import datetime

import copy
import paramparse

"""deep mdp modules"""
from input import Input
from objects import Annotations
from data import Data

from utilities import CustomLogger, SIIF, linux_path

from densecap_utilities import build_targets_densecap, build_targets_seq


class Params:
    """
    :ivar mode:
    0: build_targets_densecap
    1: build_targets_seq

    """

    def __init__(self):
        self.gpu = ''
        self.cfg = ('',)

        self.set = ''
        self.seq = ()

        self.mode = 0
        self.load = 0
        self.save = 1
        self.start = 0

        self.grid_res = (32, 32)
        self.frame_gap = 1
        self.fps = 30
        self.vis = 0

        self.interval = 1
        self.win_size = 480
        self.win_stride = 0

        self.n_proc = 1

        self.input = Input.Params(source_type=-1, batch_mode=False)

        self.data = Data.Params()
        self.ann = Annotations.Params()


def run(seq_info, n_seq, out_dir, traj_lengths_out_dir, params):
    """

    :param seq_id:
    :param start_id:
    :param end_id:
    :param n_seq:
    :param out_dir:
    :param traj_lengths_out_dir:
    :param Params params:
    :return:
    """

    seq_id, seq_suffix, start_id, end_id = seq_info

    _logger = CustomLogger.setup(__name__)

    _data = Data(params.data, _logger)

    if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
        _logger.error('Data module could not be initialized')
        return None

    subset = "training" if _data.split == 'train' else "validation"

    input_params = copy.deepcopy(params.input)  # type: Input.Params

    input_params.frame_ids = (start_id, end_id)

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

    # if not _input.read_detections():
    #     _logger.error('Detections could not be read')
    #     return False

    _frame_size = _input.frame_size
    _n_frames = _input.n_frames

    duration = float(_n_frames) / params.fps
    # _frames = _input.all_frames
    _annotations = _input.annotations  # type: Annotations
    # _detections = _input.detections  # type: Detections

    if params.vis:
        _input._read_all_frames()

    if params.mode == 0:
        n_frames = _input.n_frames
        frame_size = _input.frame_size

        vocab_annotations, traj_lengths = build_targets_densecap(n_frames,
                                                                 frame_size,
                                                                 _input.all_frames,
                                                                 _annotations,
                                                                 grid_res=params.grid_res,
                                                                 frame_gap=params.frame_gap,
                                                                 win_size=params.win_size,
                                                                 fps=params.fps,
                                                                 out_dir=out_dir,
                                                                 vis=params.vis,
                                                                 )
        mean_traj_length = np.mean(traj_lengths)
        std_traj_length = np.std(traj_lengths)
        median_traj_length = np.median(traj_lengths)
        min_traj_length = np.amin(traj_lengths)
        max_traj_length = np.amax(traj_lengths)

        print(f'\nseq traj_length: '
              f'mean: {mean_traj_length} '
              f'median: {median_traj_length} '
              f'min: {min_traj_length} '
              f'max: {max_traj_length} '
              f'std: {std_traj_length} '
              )

        traj_lengths_out_path = linux_path(traj_lengths_out_dir, f'{seq_name}.txt')
        np.savetxt(traj_lengths_out_path, np.asarray(traj_lengths, dtype=np.uint32), fmt='%d')

        duration_frame_csv_row = {
            'name': seq_name,
            'duration': duration,
            'n_frames': _n_frames,
        }

        return seq_id, traj_lengths, duration_frame_csv_row, vocab_annotations, subset
    else:
        _input._read_all_frames()
        build_targets_seq(_input.all_frames, _annotations)


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

    interval = params.interval
    if interval <= 0:
        interval = 1

    seq_info = []
    pbar = tqdm(seq_ids)
    for seq_id in pbar:

        if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
            _logger.error('Data module could not be initialized')
            return None

        start_id = 0
        seq_name = _data.seq_name
        seq_n_frames = _data.seq_n_frames

        assert seq_n_frames % interval == 0, f"interval {interval} does not divide seq_n_frames {seq_n_frames} evenly"

        win_size = params.win_size
        if win_size <= 0:
            win_size = int(_data.seq_n_frames / interval)

        win_stride = params.win_stride
        if win_stride <= 0:
            win_stride = win_size

        while True:
            abs_start_id = int(start_id * interval)

            if abs_start_id >= seq_n_frames:
                break

            end_id = start_id + win_size

            abs_end_id = int(end_id * interval)

            if abs_end_id >= seq_n_frames:
                abs_end_id = seq_n_frames
                end_id = int(abs_end_id / interval)

            suffix = f'{start_id}_{end_id}'

            seq_info.append((seq_id, suffix, abs_start_id, abs_end_id))

            print(f'{seq_name}--{suffix}: {abs_start_id} to {abs_end_id}')

            start_id += win_stride

    n_seq = len(seq_info)

    exit()

    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    out_dir = linux_path('log', f'build_targets_densecap_{timestamp}')
    os.makedirs(out_dir, exist_ok=1)

    print(f'out_dir: {out_dir}')

    traj_lengths_out_dir = linux_path(out_dir, 'traj_lengths')
    os.makedirs(traj_lengths_out_dir, exist_ok=1)
    print(f'traj_lengths_out_dir: {traj_lengths_out_dir}')

    database = {}
    duration_frame_csv_rows = []
    all_traj_lengths = []
    seq_to_traj_lengths = []

    n_trajectories = 0

    n_proc = min(params.n_proc, n_seq)

    import functools
    func = functools.partial(
        run,
        n_seq=n_seq,
        out_dir=out_dir,
        traj_lengths_out_dir=traj_lengths_out_dir,
        params=params,
    )

    if n_proc > 1:
        import multiprocessing

        print(f'running in parallel over {n_proc} processes')
        with multiprocessing.Pool(n_proc) as pool:
            results = pool.map(func, seq_info)

        results.sort(key=lambda x: x[0])
    else:
        results = []
        for seq_id in params.seq:
            result = func(seq_id)

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

    print(f'saving json with {n_seq} sequences and {n_trajectories} trajectories to: {json_path}')
    with open(json_path, 'w') as f:
        output_json_data = json.dumps(json_dict, indent=4)
        f.write(output_json_data)

    csv_path = os.path.join(out_dir, f'{set_name}_duration_frame.csv')
    print(f'saving duration_frame_csv to: {csv_path}')
    df = pd.DataFrame(duration_frame_csv_rows)
    df.to_csv(csv_path, index=False, header=False)


if __name__ == "__main__":
    main()
