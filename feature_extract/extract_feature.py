import os
import sys

sys.path.append('/home/abhineet/isl_labeling_tool/deep_mdp')

import paramparse

from input import Input
from data import Data
from objects import Annotations

from utilities import CustomLogger, linux_path


class Params:
    """
    :ivar gpu:
    :type gpu: int

    :ivar use_flow:
    :type use_flow: bool

    """

    def __init__(self):
        self.cfg = ('',)
        self.gpu = 0
        self.use_flow = 1

        self.set = ''
        self.seq = ()
        self.start_seq = 0
        self.end_seq = -1
        self.out_path = ''

        """
        number of consecutive frames from which the optical flow features are extracted
        each such set of frames contributes a single feature vector to the network
        """
        self.length = 6
        """
        this is the number of sampled frames, i.e. number of frames represented by a single feature in the dnc input
        assuming fps=30,  interval=15 frames corresponds to 0.5 seconds
        """
        self.interval = 15

        self.slide_window_size = 15
        self.slide_window_stride = 15

        self.new_size = (340, 256)

        self.input = Input.Params(source_type=-1, batch_mode=False)
        self.data = Data.Params()
        self.ann = Annotations.Params()

        self.n_proc = 1
        self.n_gpu = -1
        self.gpus = ()
        self.win_id = 'x99'
        self.pane_id = 6


def main():
    params = Params()
    paramparse.process(params)

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

    models = [('models/resnet200_anet_2016_deploy.prototxt',
               'models/resnet200_anet_2016.caffemodel',
               1.0, 0, True, 224)]

    if params.use_flow:
        models.append(('models/bn_inception_anet_2016_temporal_deploy.prototxt',
                       'models/bn_inception_anet_2016_temporal.caffemodel.v5',
                       0.2, 1, False, 224))
    else:
        params.length = 1

    if params.interval < params.length:
        params.interval = params.length

    seq_id_info_list = list(enumerate(params.seq))

    # io_info_list = []
    # for seq_id_info in seq_id_info_list:
    #     io_info = run(seq_id_info, params, n_seq)
    #     io_info_list.append(io_info)

    # gpu_id = 1

    from pyActionRec.action_classifier import ActionClassifier

    cls = ActionClassifier(models, dev_id=params.gpu)

    # cls.classify_mp(params, io_info_list)

    for seq_id_info in seq_id_info_list:
        __id, seq_id = seq_id_info

        if not _data.initialize(params.set, seq_id, 0, _logger):
            _logger.error('Data module could not be initialized')
            return None

        _input = Input(params.input, _logger)

        if not _input.initialize(_data):
            _logger.error('Input pipeline could not be initialized')
            return False

        if params.out_path:
            out_path = params.out_path
        else:
            split = 'training' if _data.split == 'train' else 'validation'
            out_path = linux_path(_input.seq_set_path, f'features_{params.length}_{params.interval}', split)

        print(f'\nseq {__id + 1} / {n_seq}: {seq_id}: {_input.seq_path}\n')
        print(f'out_path: {out_path}')

        os.makedirs(out_path, exist_ok=1)

        cls.classify((_input, out_path), length=params.length, new_size=params.new_size,
                     interval=params.interval)


if __name__ == "__main__":
    main()
