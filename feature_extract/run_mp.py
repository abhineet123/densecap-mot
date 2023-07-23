import os
import sys

sys.path.append('/home/abhineet/isl_labeling_tool/deep_mdp')

import math

import paramparse

# from multiprocessing.pool import ThreadPool
# from contextlib import closing
# import multiprocessing

# import functools
# func = functools.partial(run, params=params, n_seq=n_seq, cls=cls)
# func = functools.partial(cls.classify_mp, params=params, n_seq=n_seq)
# with closing(ThreadPool(n_proc)) as pool:
#     # with  multiprocessing.Pool(n_proc) as pool:
#     pool.map(func, seq_id_info_list)

import subprocess
import shlex

from extract_feature import Params

from data import Data

from utilities import CustomLogger


def main():
    params = Params()
    args_in = paramparse.process(params)
    print(f'args_in: {args_in}')

    n_proc = params.n_proc
    n_gpu = params.n_gpu

    if n_gpu <= 0:
        import torch
        n_gpu = torch.cuda.device_count()

    print(f'n_gpu: {n_gpu}')

    try:
        params.set = int(params.set)
    except ValueError:
        params.set = params.data.name_to_id(params.set)

    _logger = CustomLogger.setup(__name__)
    _data = Data(params.data, _logger)

    set_name = _data.sets[params.set]
    n_sequences = len(_data.sequences[set_name])

    if not params.seq:
        params.seq = tuple(range(n_sequences))

    if params.end_seq < 0:
        params.end_seq = len(params.seq) - 1

    params.seq = params.seq[params.start_seq:params.end_seq + 1]
    n_seq = len(params.seq)

    if n_proc > 1:
        print(f'running in parallel over {n_proc} processes')

    n_seq_per_proc = int(math.ceil(n_seq / n_proc))
    start_seq_id = 0
    processes = []

    excl_params = ['n_proc', 'n_gpu', 'win_id', 'pane_id']
    incl_params = [k for k in sys.argv[1:] if not any(k.startswith(p)
                                                      for p in excl_params)]
    in_arg_str = ' '.join(incl_params)

    cmd_list = []

    for proc_id in range(n_proc):
        gpu_id = (proc_id + 1) % n_gpu
        end_seq_id = min(start_seq_id + n_seq_per_proc - 1, n_seq - 1)

        # seqs = list(map(str, params.seq[start_seq_id:end_seq_id + 1]))
        # seqs_str = ','.join(seqs)

        tmux_id = f"## @ {params.win_id}:{params.pane_id}.{proc_id}"
        cmd_list.append(tmux_id)
        proc_cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python extract_feature.py {in_arg_str} ' \
            f'start_seq={start_seq_id} ' \
            f'end_seq={end_seq_id}'
        cmd_list.append(proc_cmd + '\n')

        # print(f'proc {proc_id}: {proc_cmd}')
        # args = shlex.split(proc_cmd)
        # p = subprocess.Popen(args)
        # processes.append(p)

        start_seq_id = end_seq_id + 1

    out_file = 'cmd_list.txt'
    print(f'out_file: {out_file}')
    cmd_list_str = '\n'.join(cmd_list)
    print(cmd_list_str)
    with open(out_file, 'w') as fid:
        fid.write(cmd_list_str)

    # kill_all = False
    # for p in processes:
    #     if kill_all:
    #         p.kill()
    #         p.wait()
    #
    #         continue
    #     try:
    #         p.wait()
    #     except KeyboardInterrupt:
    #         kill_all = True
    #         p.kill()
    #         p.wait()

if __name__ == "__main__":
    main()
