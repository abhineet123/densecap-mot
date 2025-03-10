import os
import sys
import logging

# script_dir = os.path.dirname(os.path.abspath(__file__))
# script_parent_dir = script_dir.replace(os.sep, '/') + '/..'
# sys.path.append(script_parent_dir)


home_path = os.path.expanduser('~')
deep_mdp_path = os.path.join(home_path, 'isl_labeling_tool', 'deep_mdp')
sys.path.append(deep_mdp_path)

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

from dnc_utilities import build_targets_densecap, build_targets_seq


class Params:
    class SlidingWindow:
        num = 0
        sample = 0
        size = 0
        stride = 0

    def __init__(self):
        self.gpu = ''
        self.cfg = ('',)

        self.set = ''
        self.seq = ()
        self.start_seq = ()
        self.end_seq = ()

        """:ivar mode:
            0: build_targets_densecap
            1: build_targets_seq
        """
        self.mode = 0

        self.load = 0
        self.save = 1
        self.start = 0

        self.no_repeat = 0

        self.grid_res = (32, 32)
        self.frame_gap = 1
        self.fps = 30
        self.vis = 0
        """
        0: absolute grid cell addresses
        1: differential grid cell addresses with separate row and column for the starting location
        2: differential grid cell addresses with combined row and column for the starting location        
        """
        self.vocab_fmt = 0
        self.max_diff = 1
        self.sample_traj = 0

        self.min_traj_len = 0
        self.fixed_traj_len = 0
        self.save_traj_lengths = 0

        self.win_size = 0

        self.n_proc = 1

        self.slide = Params.SlidingWindow()
        self.input = Input.Params(source_type=-1, batch_mode=False)
        self.data = Data.Params()
        self.ann = Annotations.Params()


def run(seq_info, sample_traj, min_traj_len, fixed_traj_len, out_dir, traj_lengths_out_dir, params: Params):
    seq_id, start_id, end_id = seq_info

    _logger = CustomLogger.setup(__name__)
    _logger.setLevel(logging.WARNING)

    _data = Data(params.data, _logger)

    if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
        _logger.error('Data module could not be initialized')
        return None

    subset = "training" if _data.split == 'train' else "validation"

    input_params = copy.deepcopy(params.input)  # type: Input.Params

    """end_id is exclusive but Input expects inclusive"""
    input_params.frame_ids = (start_id, end_id - 1)

    _input = Input(input_params, _logger)
    seq_name = _data.seq_name

    seq_name = f'{seq_name}--{start_id}_{end_id}'

    print(f'\nseq {seq_id + 1}: {seq_name}\n')

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

        vocab_annotations, traj_lengths, vocab = build_targets_densecap(
            params.vocab_fmt,
            params.max_diff,
            sample_traj,
            min_traj_len,
            fixed_traj_len,
            n_frames,
            frame_size,
            _input.all_frames,
            _annotations,
            seq_name=seq_name,
            grid_res=params.grid_res,
            frame_gap=params.frame_gap,
            win_size=params.win_size,
            fps=params.fps,
            out_dir=out_dir,
            no_repeat=params.no_repeat,
            vis=params.vis,
        )
        # mean_traj_length = np.mean(traj_lengths)
        # std_traj_length = np.std(traj_lengths)
        # median_traj_length = np.median(traj_lengths)
        # min_traj_length = np.amin(traj_lengths)
        # max_traj_length = np.amax(traj_lengths)

        # print(f'\nseq traj_length: '
        #       f'mean: {mean_traj_length} '
        #       f'median: {median_traj_length} '
        #       f'min: {min_traj_length} '
        #       f'max: {max_traj_length} '
        #       f'std: {std_traj_length} '
        #       )

        if params.save_traj_lengths:
            traj_lengths_out_path = linux_path(traj_lengths_out_dir, f'{seq_name}.txt')
            np.savetxt(traj_lengths_out_path, np.asarray(traj_lengths, dtype=np.uint32), fmt='%d')

        duration_frame_csv_row = {
            'name': seq_name,
            'duration': duration,
            'n_frames': _n_frames,
        }

        return seq_id, traj_lengths, duration_frame_csv_row, vocab_annotations, subset, vocab
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

    if params.start_seq or params.end_seq:
        assert len(params.start_seq) == len(params.end_seq), "mismatch between start_seq and end_seq lengths"
        temp_seq_ids = []
        for start_seq, end_seq in zip(params.start_seq, params.end_seq):
            if start_seq < 0:
                start_seq = 0

            if end_seq < 0:
                end_seq = len(seq_ids) - 1

            temp_seq_ids += list(seq_ids[start_seq:end_seq + 1])
        seq_ids = tuple(temp_seq_ids)

    sample = params.slide.sample
    if sample <= 0:
        sample = 1

    if params.sample_traj:
        sample_traj = sample
    else:
        sample_traj = 1

    seq_info_list = []
    pbar = tqdm(seq_ids)
    for seq_id in pbar:

        if not _data.initialize(params.set, seq_id, 0, _logger, silent=1):
            _logger.error('Data module could not be initialized')
            return None

        start_id = 0
        seq_name = _data.seq_name
        seq_n_frames = _data.seq_n_frames

        assert seq_n_frames % sample == 0, f"sample size {sample} does not divide seq_n_frames {seq_n_frames} evenly"

        win_size = params.slide.size
        if win_size <= 0:
            win_size = int(_data.seq_n_frames / sample)

        win_stride = params.slide.stride
        if win_stride <= 0:
            win_stride = win_size

        win_id = 0
        while True:
            abs_start_id = int(start_id * sample)

            if abs_start_id >= seq_n_frames or win_id >= params.slide.num > 0:
                break

            end_id = start_id + win_size

            abs_end_id = int(end_id * sample)

            if abs_end_id >= seq_n_frames:
                abs_end_id = seq_n_frames
                end_id = int(abs_end_id / sample)

            # suffix = f'{abs_start_id}_{abs_end_id}'

            seq_info_list.append((seq_id, abs_start_id, abs_end_id))

            # print(f'{seq_name}--{suffix}: {abs_start_id} to {abs_end_id}')

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

    import functools
    func = functools.partial(
        run,
        min_traj_len=params.min_traj_len,
        fixed_traj_len=params.fixed_traj_len,
        sample_traj=sample_traj,
        out_dir=out_dir,
        traj_lengths_out_dir=traj_lengths_out_dir,
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

    all_vocab = []

    for seq_id, traj_lengths, duration_frame_csv_row, vocab_annotations, subset, vocab in tqdm(
            results, desc='postprocessing results'):
        if not vocab_annotations:
            continue

        all_vocab += vocab

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

    all_vocab = sorted(list(set(all_vocab)))
    print(f'Vocabulary size: {len(all_vocab)}')

    vocab_out_path = linux_path(out_dir, 'vocab.txt')
    print(f'vocab_out_path: {vocab_out_path}')
    with open(vocab_out_path, 'w') as fid:
        fid.write('\n'.join(all_vocab))

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
