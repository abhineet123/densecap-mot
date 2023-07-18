from .action_caffe import CaffeNet
from .action_flow import FlowExtractor
from .video_proc import VideoProc
from .anet_db import Video

from tqdm import tqdm
# from .pyActionRec_utils.video_funcs import sliding_window_aggregation_func, default_fusion_func
import numpy as np
import time
# import youtube_dl
import os
import subprocess

import sys

sys.path.append('~/isl_labeling_tool/deep_mdp')

# from utilities import ImageSequenceCapture
from input import Input
from data import Data
from utilities import CustomLogger, linux_path


# get video duration
def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]


class ActionClassifier(object):
    """
    This class provides and end-to-end interface to classifying videos into activity classes
    """

    def __init__(self, models, total_norm_weights=None, score_name='', dev_id=0):
        """
        Contruct an action classifier
        Args:
            models: list of tuples in the form of
                    (model_proto, model_params, model_fusion_weight, input_type, conv_support, input_size).
                    input_type is: 0-RGB, 1-Optical flow.
                    conv_support indicates whether the network supports convolution testing, which is faster. If this is
                    not supported, we will use oversampling instead
            total_norm_weights: sum of all model_fusion_weights when normalization is wanted, otherwise use None
        """

        self.__net_vec = [CaffeNet(x[0], x[1], dev_id,
                                   input_size=(340, 256) if x[4] else None
                                   ) for x in models]
        self.__net_weights = [float(x[2]) for x in models]

        if total_norm_weights is not None:
            s = sum(self.__net_weights)
            self.__net_weights = [x / s for x in self.__net_weights]

        self.__input_type = [x[3] for x in models]
        self.__conv_support = [x[4] for x in models]

        self.__num_net = len(models)

        # the input size of the network
        self.__input_size = [x[5] for x in models]

        # whether we should prepare flow stack
        self.__need_flow = max(self.__input_type) > 0

        # the name in the proto for action classes
        self.__score_name_resnet = 'caffe.Flatten_673'
        self.__score_name_bn = 'global_pool'

        # the video downloader
        # self.__video_dl = youtube_dl.YoutubeDL(
        #     {
        #         'outtmpl': '%(id)s.%(ext)s'
        #     }
        # )

        if self.__need_flow:
            self.__flow_extractor = FlowExtractor(dev_id)

    def classify_mp(self, seq_id_info, params, n_seq, model_mask=None):

        logger = CustomLogger.setup(__name__)

        seq_id, __id = seq_id_info

        _data = Data(params.data, logger)
        if not _data.initialize(params.set, seq_id, 0, logger):
            logger.error('Data module could not be initialized')
            return None

        _input = Input(params.input, logger)

        if not _input.initialize(_data):
            logger.error('Input pipeline could not be initialized')
            return False

        if params.out_path:
            out_path = params.out_path
        else:
            split = 'training' if _data.split == 'train' else 'validation'
            out_path = linux_path(_input.seq_set_path, f'features_{params.length}_{params.interval}', split)

        print(f'\nseq {__id + 1} / {n_seq}: {_input.seq_path}\n')
        print(f'out_path: {out_path}')

        os.makedirs(out_path, exist_ok=1)

        self.classify((_input, out_path), length=params.length, new_size=params.new_size,
                      interval=params.interval, model_mask=model_mask)

        # import functools
        # func = functools.partial(self.classify, length=length, new_size=new_size, interval=interval, model_mask=model_mask)
        #
        # if params.n_proc > 1:
        #     import multiprocessing
        #     import functools
        #
        #     print(f'running in parallel over {params.n_proc} processes')
        #     pool = multiprocessing.Pool(params.n_proc)
        #     pool.map(func, io_info_list)
        # else:
        #     for io_info in io_info_list:
        #         func(io_info)

    def classify(self, io_info, length, new_size, interval, model_mask=None):
        """
        :param Input _input:
        :return:
        """

        _input, out_path = io_info

        frm_it = _input.read_iter(
            length=length,
            new_size=new_size,
            interval=interval,
        )
        n_src_files = _input.n_frames

        # duration = getLength(src_path)
        # duration_in_second = float(duration[0][15:17]) * 60 + float(duration[0][18:23])
        # info_dict = {
        #     'annotations': list(),
        #     'url': '',
        #     'duration': duration_in_second,
        #     'subset': 'testing'
        # }
        #
        # vid_info = Video('0', info_dict)
        # # update dummy video info...
        #
        # vid_info.path = src_path
        # video_proc = VideoProc(vid_info)
        # video_proc.open_video(True)
        # n_src_files = video_proc._frame_count
        #
        # # here we use interval of 30, roughly 1FPS
        # frm_it = video_proc.frame_iter(
        #     timely=True,
        #     ignore_err=True,
        #     interval=interval_t,
        #     length=length,
        #     new_size=new_size)

        all_features = {'resnet': np.empty(shape=(0, 2048)), 'bn': np.empty(shape=(0, 1024))}
        # all_start = time.time()

        cnt = 0

        # process model mask
        mask = [True] * self.__num_net
        n_model = self.__num_net
        if model_mask is not None:
            for i in range(len(model_mask)):
                mask[i] = model_mask[i]
                if not mask[i]:
                    n_model -= 1

        n_frm_stacks = int(n_src_files / max(interval, length))

        print(f'n_src_files: {n_src_files}')
        print(f'n_frm_stacks: {n_frm_stacks}')

        for frm_stack in tqdm(frm_it, ncols=100, total=n_frm_stacks):

            # start = time.time()
            cnt += 1

            flow_stack = None

            """each frame stack is processed twice – once to extract the resnet features for the first frame 
            and next to extract the pairwise optical flow for the entire stack followed by 
            some kind of global pooling as suggested by the feature name"""
            for net, run, in_type, conv_support, net_input_size in \
                    zip(self.__net_vec, mask, self.__input_type, self.__conv_support, self.__input_size):
                if not run:
                    continue

                frame_size = (340 * net_input_size / 224, 256 * net_input_size / 224)

                if in_type == 0:
                    # RGB input
                    # TODO for now we only sample one frame w/o applying mean-pooling
                    all_features['resnet'] = np.concatenate(
                        (all_features['resnet'], net.predict_single_frame(frm_stack[:1], self.__score_name_resnet,
                                                                          over_sample=not conv_support,
                                                                          frame_size=None if net_input_size == 224
                                                                          else frame_size
                                                                          )), axis=0)
                elif in_type == 1:
                    # Flow input
                    if flow_stack is None:
                        # Extract flow if necessary
                        # we disabled spatial data aug
                        # the size for flow frames are 224 x 224, hard coded
                        flow_frame_size = (224, 224)
                        flow_stack = self.__flow_extractor.extract_flow(frm_stack, flow_frame_size)

                    """ store all the optical flow features"""
                    # all_features['bn'] = np.concatenate((all_features['bn'], np.squeeze(
                    # net.predict_single_flow_stack(flow_stack, self.__score_name_bn,
                    #                   over_sample=not conv_support))), axis=0)

                    """store only the optical flow feature for the center crop"""
                    bn_aug = np.squeeze(net.predict_single_flow_stack(flow_stack, self.__score_name_bn,
                                                                      over_sample=False))
                    # over_sample=not conv_support))
                    # bn_aug = np.squeeze(bn_aug)
                    # bn_center = bn_aug[5]
                    bn_center = bn_aug
                    bn_center = np.reshape(bn_center, (1, bn_center.shape[0]))
                    all_features['bn'] = np.concatenate((all_features['bn'], bn_center), axis=0)

            # end = time.time()
            # elapsed = end - start
            # print("frame sample {}: {} second".format(cnt, elapsed))

        resnet_out_fname = _input.seq_name + "_resnet.npy"
        bn_out_fname = _input.seq_name + "_bn.npy"

        resnet_out_path = os.path.join(out_path, resnet_out_fname)
        bn_out_path = os.path.join(out_path, bn_out_fname)

        resnet_out_shape = all_features['resnet'].shape
        bn_out_shape = all_features['bn'].shape

        print(f'saving resnet features of shape {resnet_out_shape} to: {resnet_out_path}')
        print(f'saving bn features of shape {bn_out_shape} to: {bn_out_path}')

        np.save(resnet_out_path, all_features['resnet'])
        np.save(bn_out_path, all_features['bn'])

        # return all_features

    def _classify_from_url(self, url, model_mask):
        """
        This function classify an video based on input video url
        It will first use Youtube-dl to download the video. Then will do classification on the downloaded file
        Returns:
            cls: classification scores
            all_features: RGB ResNet feature and Optical flow BN Inception feature in a list
        """

        file_info = self.__video_dl.extract_info(url)  # it also downloads the video file
        filename = file_info['id'] + '.' + file_info['ext']

        scores, all_features, total_time = self._classify_from_file(filename, model_mask)
        import os
        os.remove(filename)
        return scores, all_features, total_time
