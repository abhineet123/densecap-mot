"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import pickle
import time

import torch
import numpy as np
from torch.utils.data import Dataset


class ANetTestDataset(Dataset):
    def __init__(self,
                 image_path,
                 slide_window_size,
                 dur_file,
                 dataset,
                 sampling_sec,
                 text_proc,
                 raw_data,
                 split,
                 learn_mask,
                 sample_list_dir,
                 enable_flow
                 ):
        super(ANetTestDataset, self).__init__()

        self.split = split
        split_path = os.path.join(image_path, self.split)
        self.slide_window_size = slide_window_size
        self.learn_mask = learn_mask
        self.enable_flow = enable_flow

        self.sample_list_dir = sample_list_dir

        self.frame_to_second = {}
        self.sampled_frames = {}
        self.fps = {}
        self.vid_dur = {}
        self.vid_frame = {}

        with open(dur_file) as f:
            if dataset == 'anet':
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                    self.frame_to_second[vid_name] = float(vid_dur) * int(
                        float(vid_frame) * 1. / int(float(vid_dur)) * sampling_sec) * 1. / float(vid_frame)
                self.frame_to_second['_0CqozZun3U'] = sampling_sec  # a missing video in anet
            elif dataset == 'yc2':
                import math
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                    self.frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)  # for yc2
            elif dataset.startswith('MNIST_MOT_RGB'):
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                    self.frame_to_second[vid_name] = float(vid_dur) * int(
                        float(vid_frame) * 1. / int(float(vid_dur)) * sampling_sec) * 1. / float(vid_frame)

                    self.vid_dur[vid_name] = float(vid_dur)
                    self.vid_frame[vid_name] = float(vid_frame)
                    self.fps[vid_name] = float(vid_frame) / float(vid_dur)
                    self.sampled_frames[vid_name] = int(self.fps[vid_name] * sampling_sec)
            else:
                raise NotImplementedError(f"Unsupported dataset: {dataset}")

        sentences_dict_path = os.path.join(self.sample_list_dir, f"{self.split}_sentences_dict.pkl")
        if os.path.isfile(sentences_dict_path):
            print(f'loading sentences_dict from: {sentences_dict_path}')
            with open(sentences_dict_path, 'rb') as f:
                sentences_dict = pickle.load(f)
            test_sentences = sentences_dict['test_sentences']
            sentence_idx = sentences_dict['sentence_idx']
            self.sample_list = sentences_dict['sample_list']
        else:
            self.sample_list = []
            test_sentences = []
            for vid, val in raw_data.items():
                annotations = val['annotations']
                if val['subset'] != self.split:
                    continue

                # file_path = os.path.join(split_path, vid + '_bn.npy')
                # assert os.path.isfile(file_path), "file does not exist: {}".format(file_path)
                if self.enable_flow:
                    if '--' in vid:
                        feat_name, vid_frame_ids = vid.split('--')
                        vid_frame_ids = tuple(map(int, vid_frame_ids.split('_')))

                        start_id, end_id = vid_frame_ids

                        n_subseq_frames = end_id - start_id
                        assert n_subseq_frames == self.vid_frame[vid], \
                            f'n_subseq_frames mismatch: {n_subseq_frames}, {self.vid_frame[vid]}'

                        feat_start_id, feat_end_id = int(start_id / self.sampled_frames[vid]), int(
                            end_id / self.sampled_frames[vid])

                        feat_frame_ids = (feat_start_id, feat_end_id)
                    else:
                        feat_name = vid
                        feat_frame_ids = None

                    video_prefix = os.path.join(split_path, feat_name)
                else:
                    """assume that each npy file contains features only for one subsequence"""
                    video_prefix = os.path.join(split_path, vid)
                    feat_frame_ids = None

                self.sample_list.append((video_prefix, feat_frame_ids))

                for ind, ann in enumerate(annotations):
                    ann['sentence'] = ann['sentence'].strip()
                    test_sentences.append(ann['sentence'])

            test_sentences = list(map(text_proc.preprocess, test_sentences))
            sentence_idx = text_proc.numericalize(text_proc.pad(test_sentences),
                                                  device=None)
            sentences_dict = dict(
                test_sentences=test_sentences,
                sentence_idx=sentence_idx,
                sample_list=self.sample_list,
            )
            print(f'saving sentences_dict to: {sentences_dict_path}')
            os.makedirs(self.sample_list_dir, exist_ok=1)
            with open(sentences_dict_path, 'wb') as f:
                pickle.dump(sentences_dict, f)

            if sentence_idx.nelement() != 0 and len(test_sentences) != 0:
                if sentence_idx.size(0) != len(test_sentences):
                    raise Exception("Error in numericalizing sentences")

        idx = 0
        for vid, val in raw_data.items():
            if val['subset'] != self.split:
                continue
            for ann in val['annotations']:
                ann['sentence_idx'] = sentence_idx[idx]
                idx += 1

        print('total number of samples (unique videos): {}'.format(
            len(self.sample_list)))
        print('total number of sentences: {}'.format(len(test_sentences)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        video_prefix, feat_frame_ids = self.sample_list[index]

        if self.enable_flow:

            start = time.time()
            if feat_frame_ids is not None:
                feat_start_id, feat_end_id = feat_frame_ids

                resnet_feat = np.load(video_prefix + '_resnet.npy', mmap_mode='r')[feat_start_id:feat_end_id, ...]
                bn_feat = np.load(video_prefix + '_bn.npy', mmap_mode='r')[feat_start_id:feat_end_id, ...]

                resnet_feat = np.array(resnet_feat)
                bn_feat = np.array(bn_feat)

            else:
                resnet_feat = np.load(video_prefix + '_resnet.npy')
                bn_feat = np.load(video_prefix + '_bn.npy')

            end = time.time()

            resnet_feat = torch.from_numpy(resnet_feat).float()
            bn_feat = torch.from_numpy(bn_feat).float()


            if self.learn_mask:
                img_feat = torch.from_numpy(np.zeros((self.slide_window_size,
                                                      resnet_feat.size(1) + bn_feat.size(1)))).float()
                torch.cat((resnet_feat, bn_feat), dim=1,
                          out=img_feat[:min(bn_feat.size(0), self.slide_window_size)])
            else:
                img_feat = torch.cat((resnet_feat, bn_feat), 1)
            end2 = time.time()

        else:
            start = time.time()
            img_feat_np = np.load(video_prefix + '.npy', mmap_mode='r')
            end = time.time()

            img_feat = torch.from_numpy(img_feat_np).float()
            end2 = time.time()

        load_t = (end - start) * 1000
        torch_t = (end2 - end) * 1000

        total_frames = img_feat.size(0)

        return img_feat, total_frames, video_prefix, feat_frame_ids, load_t, torch_t


def anet_test_collate_fn(batch_lst):
    start = time.time()

    img_feat = batch_lst[0][0]

    batch_size = len(batch_lst)

    img_batch = torch.zeros(batch_size, img_feat.size(0), img_feat.size(1))

    frame_length = torch.zeros(batch_size, dtype=torch.int)

    video_prefix = []
    feat_frame_ids_all = []

    batch_load_t = 0
    batch_torch_t = 0

    for batch_idx in range(batch_size):
        img_feat, total_frames, vid, feat_frame_ids, load_t, torch_t = batch_lst[batch_idx]

        batch_load_t += load_t
        batch_torch_t += torch_t

        img_batch[batch_idx, :] = img_feat
        frame_length[batch_idx] = total_frames
        video_prefix.append(vid)
        feat_frame_ids_all.append(feat_frame_ids)

    end = time.time()
    collate_t = (end - start) * 1000
    times = (batch_load_t, batch_torch_t, collate_t)

    return img_batch, frame_length, video_prefix, feat_frame_ids_all, times
