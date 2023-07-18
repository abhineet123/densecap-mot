import os
import sys

sys.path.append('/home/abhineet/isl_labeling_tool/deep_mdp')

import paramparse

from extract_feature_params import ExtractFeatureParams

from input import Input
from data import Data

from utilities import CustomLogger, linux_path


def main():
    params = ExtractFeatureParams()
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
