"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import sys
import os
import json
import numpy as np
import glob
from collections import defaultdict
import math
import time
import multiprocessing
import functools

import pickle
from random import shuffle
from tqdm import tqdm

import torch
import torchtext
from torch.utils.data import Dataset

from densecap_data.utils import segment_iou


# dataloader for training
class ANetDataset(Dataset):
    def __init__(self, image_path,
                 n_vids,
                 splits,
                 slide_window_size,
                 dur_file, kernel_list, text_proc, raw_data,
                 pos_thresh, neg_thresh, stride_factor, dataset,
                 sampling_sec,
                 save_samplelist,
                 load_samplelist,
                 sample_list_dir):
        super(ANetDataset, self).__init__()

        # n_vids = len(raw_data)

        split_paths = []
        for split_dev in splits:
            split_paths.append(os.path.join(image_path, split_dev))

        self.slide_window_size = slide_window_size
        self.pos_thresh = pos_thresh
        self.neg_thresh = neg_thresh
        self.raw_data = raw_data

        self.save_samplelist = save_samplelist
        self.sample_list_dir = sample_list_dir
        self.sample_list_parent_dir = os.path.dirname(self.sample_list_dir)

        self.n_vids = n_vids
        self.splits = splits
        self.split_paths = split_paths

        self.samples_loaded = False

        self.sample_list = []  # list of list for data samples
        sample_dict = defaultdict(list)

        if load_samplelist:
            if os.path.isdir(sample_list_dir):
                print(f'loading samples from: {sample_list_dir}')
                pkl_files = glob.glob(os.path.join(sample_list_dir, '*.pkl'), recursive=False)
                n_pkl_files = len(pkl_files)

                if n_pkl_files != n_vids:
                    print(f"mismatch between n_vids: {n_vids} and n_pkl_files: {n_pkl_files}")
                else:
                    pkl_files = sorted(pkl_files)
                    pbar = tqdm(pkl_files, ncols=120)
                    for pkl_file in pbar:
                        with open(pkl_file, 'rb') as f:
                            vid_sample_list = pickle.load(f)
                        self.sample_list += vid_sample_list
                        vid = os.path.splitext(os.path.basename(pkl_file))[0]

                        # vid_from_samples = [os.path.basename(sample[0]) for sample in vid_sample_list]
                        # assert all(vid_from_sample == vid for vid_from_sample in vid_from_samples), \
                        #     "vid name mismatch"

                        sample_dict[vid] = [sample[1:] for sample in vid_sample_list]

                        pbar.set_description(f'\t{vid:50s}: {len(sample_dict[vid]):5d} samples')

                    print(f'loaded {len(self.sample_list)} samples for {len(sample_dict)} videos')

                    self.samples_loaded = True
                    return

            if load_samplelist == 2:
                raise AssertionError(f'nonexistent sample_list_dir: {sample_list_dir}')

            print(f'sample list path not found so continuing with sample generation: {sample_list_dir}')

        if save_samplelist:
            assert sample_list_dir is not None, "sample list dir must be provided"
            print(f'saving sample list to {sample_list_dir}')

        sentences_dict_path = os.path.join(self.sample_list_parent_dir, f"{self.splits[0]}_sentences_dict.pkl")
        if os.path.isfile(sentences_dict_path):
            print(f'loading sentences_dict from: {sentences_dict_path}')
            with open(sentences_dict_path, 'rb') as f:
                sentences_dict = pickle.load(f)
            train_sentences = sentences_dict['train_sentences']
            sentence_idx = sentences_dict['sentence_idx']
        else:
            train_sentences = []
            for vid, val in self.raw_data.items():
                annotations = val['annotations']
                if val['subset'] in splits:
                    for ind, ann in enumerate(annotations):
                        ann['sentence'] = ann['sentence'].strip()
                        train_sentences.append(ann['sentence'])

            train_sentences = list(map(text_proc.preprocess, train_sentences))
            sentence_idx = text_proc.numericalize(text_proc.pad(train_sentences), device=None)  # put in memory
            if sentence_idx.size(0) != len(train_sentences):
                raise Exception("Error in numericalizing sentences")
            sentence_idx_np = sentence_idx.numpy()
            sentence_idx = sentence_idx_np.tolist()

            sentences_dict = dict(train_sentences=train_sentences, sentence_idx=sentence_idx)
            print(f'saving sentences_dict to: {sentences_dict_path}')
            os.makedirs(self.sample_list_parent_dir, exist_ok=1)
            with open(sentences_dict_path, 'wb') as f:
                pickle.dump(sentences_dict, f)

        idx = 0
        for vid, val in self.raw_data.items():
            for split, split_path in zip(splits, split_paths):
                if val['subset'] != split:
                    continue

                for ann in val['annotations']:
                    ann['sentence_idx'] = sentence_idx[idx]
                    idx += 1

        # print(f'size of the sentence block variable ({splits}): {sentence_idx.size()}')

        """generate anchors"""
        anc_len_lst = []
        anc_cen_lst = []
        for i in range(0, len(kernel_list)):
            kernel_len = kernel_list[i]

            # anc_stride = math.ceil(kernel_len / stride_factor)
            anc_stride = math.ceil(kernel_len / stride_factor)
            """
            equally spaced anchor centers so the first one starts at first frame and last one ends at the 
            last frame in the temporal window
            seem to be in units of frames rather than seconds
            """
            anc_cen = np.arange(
                float(kernel_len / 2.),
                float(slide_window_size + 1 - kernel_len / 2.),
                anc_stride
            )

            anc_len = np.full(anc_cen.shape, kernel_len)
            anc_len_lst.append(anc_len)
            anc_cen_lst.append(anc_cen)

        self.anc_len_all = np.hstack(anc_len_lst)
        self.anc_cen_all = np.hstack(anc_cen_lst)

        """frame_to_second is a single factor that, when multiplied with the frame ID, 
        gives the corresponding temporal location in seconds"""
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
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                    self.frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)  # for yc2
            elif dataset.startswith('MNIST_MOT_RGB'):
                for line in f:
                    vid_name, vid_dur, vid_frame = [l.strip() for l in line.split(',')]
                    # vid_fps = float(vid_frame) / float(vid_dur)
                    # sampling_frames = math.ceil(vid_fps * sampling_sec)
                    # frame_to_second[vid_name] = float(vid_dur) * sampling_frames / float(vid_frame)
                    self.vid_dur[vid_name] = float(vid_dur)
                    self.vid_frame[vid_name] = float(vid_frame)
                    self.fps[vid_name] = float(vid_frame) / float(vid_dur)
                    self.sampled_frames[vid_name] = int(self.fps[vid_name] * sampling_sec)

                    self.frame_to_second[vid_name] = float(vid_dur) * math.ceil(
                        float(vid_frame) * 1. / float(vid_dur) * sampling_sec) * 1. / float(vid_frame)
            else:
                raise NotImplementedError(f'Unsupported dataset: {dataset}')

            print(f'total number of annotations: {len(train_sentences)}')

    def get_samples(self, n_proc):
        pos_anchor_stats = []
        neg_anchor_stats = []
        print(f"generating samples for {self.splits} set")

        missing_prop = 0
        vid_idx = 0

        out_txt_dir = os.path.join(self.sample_list_dir, 'log')
        os.makedirs(out_txt_dir, exist_ok=1)

        vid_info_list = []
        for vid, val in tqdm(self.raw_data.items(), desc='generating vid_info_list', ncols=100, total=self.n_vids):
            annotations = val['annotations']

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

            for split, split_path in zip(self.splits, self.split_paths):
                if val['subset'] != split:
                    continue

                video_prefix = os.path.join(split_path, feat_name)

                # n_feat_frames = int(9600 / 15)

                resnet_feat_path = video_prefix + '_resnet.npy'
                assert os.path.isfile(resnet_feat_path), f"nonexistent resnet_feat_path: {resnet_feat_path}"

                bn_feat_path = video_prefix + '_bn.npy'
                assert os.path.isfile(bn_feat_path), f"nonexistent bn_feat_path: {bn_feat_path}"

                resnet_feat = np.load(resnet_feat_path)
                bn_feat = np.load(bn_feat_path)

                n_feat_frames = bn_feat.shape[0]

                assert resnet_feat.shape[0] == n_feat_frames, 'resnet and bn feature frames mismatch'

                if feat_frame_ids is None:
                    feat_frame_ids = (0, n_feat_frames)
                else:
                    feat_start_id, feat_end_id = feat_frame_ids
                    n_feat_frames = feat_end_id - feat_start_id

                assert self.slide_window_size >= n_feat_frames, \
                    f"n_feat_frames: {n_feat_frames} exceeds slide_window_size: {self.slide_window_size}"

                vid_info = (
                    annotations, vid, vid_idx, video_prefix, feat_frame_ids, n_feat_frames, self.frame_to_second[vid])
                vid_info_list.append(vid_info)
                vid_idx += 1

        n_vids = len(vid_info_list)
        assert n_vids == self.n_vids, "n_vids mismatch"

        if self.save_samplelist:
            print(f'saving samples to: {self.sample_list_dir}')
            os.makedirs(self.sample_list_dir, exist_ok=1)

        print('matching anchors to ground truth segments')
        print(f'out_txt_dir: {out_txt_dir}')

        if n_proc > 1:
            print(f'running in parallel over {n_proc} processes')
            func = functools.partial(
                _get_pos_neg,
                n_vids=n_vids,
                slide_window_size=self.slide_window_size,
                anc_len_all=self.anc_len_all,
                anc_cen_all=self.anc_cen_all,
                pos_thresh=self.pos_thresh,
                neg_thresh=self.neg_thresh,
                save_samplelist=self.save_samplelist,
                sample_list_dir=self.sample_list_dir,
                out_txt_dir=out_txt_dir,
            )
            with multiprocessing.Pool(n_proc) as pool:
                results = pool.map(func, vid_info_list)
        else:
            results = [None] * n_vids

            for vid_idx, vid_info in enumerate(vid_info_list):
                results[vid_idx] = _get_pos_neg(
                    vid_info,
                    n_vids=n_vids,
                    slide_window_size=self.slide_window_size,
                    anc_len_all=self.anc_len_all,
                    anc_cen_all=self.anc_cen_all,
                    pos_thresh=self.pos_thresh,
                    neg_thresh=self.neg_thresh,
                    save_samplelist=self.save_samplelist,
                    sample_list_dir=self.sample_list_dir,
                    out_txt_dir=out_txt_dir,
                )

        vid_counter = 0

        for result in results:
            if result is None:
                continue

            vid_counter += 1
            sample_list, video_prefix, vid_frame_ids, n_feat_frames, pos_seg, neg_seg, is_missing, n_pos_seg = result

            self.sample_list += sample_list

            """Only used to maintain account of any GT segments that failed to associate with any of the anchor 
            segments probably as an indicator of incorrectly chosen anchors since the latter must be chosen 
            to cover all the GT segments with sufficient numbers and at least 1 for each Otherwise there is simply no 
            way to use those missing GT segments in trainings"""
            missing_prop += is_missing

            pos_anchor_stats.append(n_pos_seg)
            n_neg_seg = len(neg_seg)
            neg_anchor_stats.append(n_neg_seg)

        avg_pos_anc = np.mean(pos_anchor_stats)
        avg_neg_anc = np.mean(neg_anchor_stats)

        print(f'total number of {self.splits} videos: {vid_counter}')
        print(f'total number of {self.splits} samples (unique segments): {len(self.sample_list)}')
        print(f'total number of missing annotations: {missing_prop}')
        print(f'avg pos anc: {avg_pos_anc:.2f} avg neg anc: {avg_neg_anc:.2f}')

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        if len(self.sample_list[index]) == 6:
            video_prefix, feat_frame_ids, pos_seg, sentence, neg_seg, total_frame = self.sample_list[index]
        else:
            video_prefix, pos_seg, sentence, neg_seg, total_frame = self.sample_list[index]
            feat_frame_ids = (0, total_frame)

        sentence = torch.from_numpy(np.asarray(sentence))

        feat_start_id, feat_end_id = feat_frame_ids

        start = time.time()
        resnet_feat = np.load(video_prefix + '_resnet.npy', mmap_mode='r')[feat_start_id:feat_end_id, ...]
        bn_feat = np.load(video_prefix + '_bn.npy', mmap_mode='r')[feat_start_id:feat_end_id, ...]

        resnet_feat = np.array(resnet_feat)
        bn_feat = np.array(bn_feat)

        end = time.time()
        load_t = (end - start) * 1000

        resnet_feat = torch.from_numpy(resnet_feat).float()
        bn_feat = torch.from_numpy(bn_feat).float()

        img_feat = torch.from_numpy(np.zeros(
            (self.slide_window_size, resnet_feat.size(1) + bn_feat.size(1)),
        )).float()

        """simply ignoring features from frames after slide_window_size"""
        torch.cat((resnet_feat, bn_feat), dim=1,
                  out=img_feat[:min(total_frame, self.slide_window_size)])

        end2 = time.time()
        torch_t = (end2 - end) * 1000

        return pos_seg, sentence, neg_seg, img_feat, load_t, torch_t


def anet_collate_fn(batch_lst):
    start = time.time()

    sample_each = 10  # TODO, hard coded
    pos_seg, sentence, neg_seg, img_feat, load_t, torch_t = batch_lst[0]

    batch_size = len(batch_lst)

    sentence_batch = torch.from_numpy(np.ones((batch_size, sentence.size(0)), dtype='int64')).long()
    img_batch = torch.from_numpy(np.zeros((batch_size,
                                           img_feat.size(0),
                                           img_feat.size(1)))).float()
    tempo_seg_pos = torch.from_numpy(np.zeros((batch_size, sample_each, 4))).float()
    tempo_seg_neg = torch.from_numpy(np.zeros((batch_size, sample_each, 2))).float()

    batch_load_t = 0
    batch_torch_t = 0

    for batch_idx in range(batch_size):
        pos_seg, sentence, neg_seg, img_feat, load_t, torch_t = batch_lst[batch_idx]

        batch_load_t += load_t
        batch_torch_t += torch_t

        img_batch[batch_idx, :] = img_feat

        pos_seg_tensor = torch.from_numpy(np.asarray(pos_seg)).float()

        sentence_batch[batch_idx] = sentence.data

        # sample positive anchors
        perm_idx = torch.randperm(len(pos_seg))
        if len(pos_seg) >= sample_each:
            tempo_seg_pos[batch_idx, :, :] = pos_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_seg_pos[batch_idx, :len(pos_seg), :] = pos_seg_tensor
            """Uniformly sample anchors to fill in the excess needed for the defined number of samples"""
            idx = torch.multinomial(torch.ones(len(pos_seg)), sample_each - len(pos_seg), True)
            tempo_seg_pos[batch_idx, len(pos_seg):, :] = pos_seg_tensor[idx]

        # sample negative anchors
        neg_seg_tensor = torch.from_numpy(np.asarray(neg_seg)).float()
        perm_idx = torch.randperm(len(neg_seg))
        if len(neg_seg) >= sample_each:
            tempo_seg_neg[batch_idx, :, :] = neg_seg_tensor[perm_idx[:sample_each]]
        else:
            tempo_seg_neg[batch_idx, :len(neg_seg), :] = neg_seg_tensor
            idx = torch.multinomial(torch.ones(len(neg_seg)),
                                    sample_each - len(neg_seg), True)
            tempo_seg_neg[batch_idx, len(neg_seg):, :] = neg_seg_tensor[idx]

    end = time.time()
    collate_t = (end - start) * 1000

    times = (batch_load_t, batch_torch_t, collate_t)

    return img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch, times


def get_vocab_and_sentences(dataset_file, splits, sample_list_path):
    # train_sentences = []
    train_val_sentences = []

    print(f'loading dataset_file: {dataset_file}')
    with open(dataset_file, 'r', encoding='utf-8') as data_file:
        data_all = json.load(data_file)
    raw_data = data_all['database']

    sample_list_dir = os.path.dirname(sample_list_path)

    sentences_dict_paths = [os.path.join(sample_list_dir, f"{split}_sentences_dict.pkl") for split in splits]
    n_train_videos = n_val_videos = 0
    # n_train_sentences = n_val_sentences = 0

    nsentence = dict(
        training=0,
        validation=0,
    )
    sentence_lengths = []
    max_sentence_length = 0

    for vid, val in tqdm(raw_data.items(), desc='getting train_val_sentences', ncols=100):
        anns = val['annotations']
        split = val['subset']
        if split == 'training':
            n_train_videos += 1
        else:
            n_val_videos += 1

        if split in ['training', 'validation']:
            for ind, ann in enumerate(anns):
                ann['sentence'] = ann['sentence'].strip()
                sentence_length = len(ann['sentence'].split(' '))
                sentence_lengths.append(sentence_length)

                if sentence_length > max_sentence_length:
                    max_sentence_length = sentence_length
                # if split == "training":
                #     train_sentences.append(ann['sentence'])
                train_val_sentences.append(ann['sentence'])
                nsentence[split] += 1

    print(f'max_sentence_length: {max_sentence_length}')

    sentence_lengths_path = os.path.join(sample_list_dir, f"sentence_lengths.txt")
    os.makedirs(sample_list_dir, exist_ok=1)

    print(f'sentence_lengths_path: {sentence_lengths_path}')

    sentence_lengths = list(map(str, sentence_lengths))
    with open(sentence_lengths_path, 'w') as fid:
        fid.write('\n'.join(sentence_lengths))

    print(f'# of training videos: {n_train_videos}')
    print('# of training sentences {}'.format(nsentence['training']))

    print(f'# of validation videos: {n_val_videos}')
    print('# of validation sentences {}'.format(nsentence['validation']))

    # if all(os.path.isfile(sentences_dict_path) for sentences_dict_path in sentences_dict_paths):
    #     print(f'ignoring annoying text_proc since sentences_dict can be loaded')
    #     # with open(sentences_dict_path, 'rb') as f:
    #     #     sentences_dict = pickle.load(f)
    #     # train_sentences = sentences_dict['train_sentences']
    #     # sentence_idx = sentences_dict['sentence_idx']
    #     text_proc = None
    # else:
    # build vocab and tokenized sentences
    try:
        Field = torchtext.data.Field
    except AttributeError:
        Field = torchtext.legacy.data.Field

    text_proc = Field(
        sequential=True, init_token='<init>',
        eos_token='<eos>', tokenize='spacy',
        lower=True, batch_first=True,
        fix_length=max_sentence_length)

    # sentences_proc = list(map(text_proc.preprocess, train_sentences)) # build vocab on train only
    """divide sentences into words to have a list of list of words or tokens as they're called"""
    sentences_proc_path = os.path.join(sample_list_dir, f"sentences_proc.pkl")
    if os.path.isfile(sentences_proc_path):
        print(f'loading sentences_proc from {sentences_proc_path}')
        with open(sentences_proc_path, 'rb') as f:
            sentences_proc = pickle.load(f)
    else:
        print('text_proc.preprocess')
        sentences_proc = list(map(text_proc.preprocess, train_val_sentences))
        print(f'saving sentences_proc to {sentences_proc_path}')
        with open(sentences_proc_path, 'wb') as f:
            pickle.dump(sentences_proc, f)

    print('building vocab')
    text_proc.build_vocab(sentences_proc, min_freq=5)
    print(f'# of words in the vocab: {len(text_proc.vocab)}')

    return text_proc, raw_data, n_train_videos, n_val_videos


def _get_pos_neg(vid_info,
                 n_vids, slide_window_size, anc_len_all,
                 anc_cen_all, pos_thresh, neg_thresh,
                 save_samplelist, sample_list_dir, out_txt_dir):
    """
    try to find a matching anchor for every single GT segment by choosing the anchor with the maximum temporal IOU
    with each GT segment and hoping that each of the latter has at least one anchor with IOU > 0.7
    also find anchor segments that can be classified as negative as being those whose temporal IOU
    with every single GT segment  < 0.3

    """
    annotations, vid, vid_idx, video_prefix, vid_frame_ids, n_frames, sampling_sec = vid_info

    print(f'\nvideo {vid_idx + 1} / {n_vids}: {vid}\n')

    window_start = 0
    window_end = slide_window_size
    window_start_t = window_start * sampling_sec
    window_end_t = window_end * sampling_sec

    """
    indexed by GT seg - list of all matching anc seg for each GT seg
    each GT segment has its own positive anchor segments but all of them have the same negative anchor segments
    since the latter must be negative with respect to all the GT segments to qualify as negative
     
    note that we do not train on the actual segments but only on the matching and nonmatching (and therefore positive 
    and negative) anchors
    however, we do consider the centre and length offsets of the actual segment from the corresponding anchor 
    to make the output more precise
    """
    pos_seg = defaultdict(list)
    n_annotations = len(annotations)
    n_anc = anc_len_all.shape[0]

    neg_overlap = [0] * n_anc
    ann_overlap = [(0, None)] * n_annotations

    anc_to_ann = [(0, None)] * n_anc

    # pos_collected = [False] * n_anc
    anc_iter = range(n_anc)

    # if not is_parallel:
    anc_iter = tqdm(anc_iter, ncols=100)

    anc_len_to_n_pos = defaultdict(int)
    gt_len_to_count = defaultdict(int)

    total_n_pos = 0
    max_seg_len = 0

    for anc_idx in anc_iter:
        """
        anchor centres and length are in units of frames - sampled rather than original
        """
        anc_cen = anc_cen_all[anc_idx]
        anc_len = anc_len_all[anc_idx]

        # if not is_parallel:
        # anc_iter.set_description(f'anc_len: {anc_len}: n_pos: {anc_len_to_n_pos[anc_len]} / {total_n_pos}')

        potential_matches = []
        for ann_idx, ann in enumerate(annotations):
            seg = ann['segment']
            seg_id = ann['id']

            """
            Something weird going on here since anchor centres and lengths are in units of frames but the raw
            GT segments are definitely in units of seconds and dividing these by sampling_sec cannot convert these 
            into frames as far as one can see
            turns out that it can be done as long as the frames are sampled frames rather than the original as is 
            indeed the case here
            """
            gt_start_raw, gt_end_raw = seg

            gt_start = gt_start_raw / sampling_sec
            gt_end = gt_end_raw / sampling_sec

            if gt_start > gt_end:
                gt_start, gt_end = gt_end, gt_start

            gt_cen = (gt_end + gt_start) / 2.
            gt_len = gt_end - gt_start

            if anc_idx == 0:
                gt_len_int = int(round(gt_len))
                if gt_len_int > max_seg_len:
                    max_seg_len = gt_len
                    # print(f'\ngt_len: {gt_len} id: {seg_id}\n')

                gt_len_to_count[gt_len_int] += 1

            anc_end = anc_cen + anc_len / 2.

            if anc_end > n_frames:
                continue

            anc_start = anc_cen - anc_len / 2.

            if window_start_t > gt_start_raw or window_end_t + sampling_sec * 2 < gt_end_raw:
                continue

            """iou between anchor and GT"""
            overlap = segment_iou(np.array([gt_start, gt_end]), np.array([[anc_start, anc_end]]))

            if isinstance(overlap, np.ndarray):
                overlap = float(overlap.item())

            """maximum overlap between this anchor segment and any GT segment"""
            neg_overlap[anc_idx] = max(overlap, neg_overlap[anc_idx])

            """maximum overlap between this GT segment segment and any anchor segment"""
            if overlap > ann_overlap[ann_idx][0]:
                ann_overlap[ann_idx] = (overlap, anc_idx)

            if overlap < pos_thresh:
                """ineligible positive sample"""
                continue

            len_offset = float(math.log(gt_len / anc_len))
            cen_offset = float((gt_cen - anc_cen) / anc_len)

            potential_match = (ann_idx, anc_idx,
                               overlap, len_offset, cen_offset,
                               ann['sentence_idx'])

            potential_matches.append(potential_match)

            # pos_collected[anc_idx] = True

        """sort the potential matches by overlap so that each anchor is matched to the maximum overlapping GT 
        that has remained unmatched so far"""
        potential_matches = sorted(potential_matches, key=lambda x: -x[2])

        """only one pos anchor segment for each GT segment"""
        filled = False
        for item in potential_matches:

            _ann_idx, _anc_idx, _overlap, _len_offset, _cen_offset, _sentence_idx = item

            """find the first matching GT segment that is not already in the list of positive GT segments 
            and add it there"""

            if _ann_idx in pos_seg:
                continue

            filled = True
            """
            the only reason why pos_seg is a dictionary of lists instead of a simple dictionary of matching 
            anchor infos seems to be that it is a defaultdict of lists which in turn seems to be entirely a convenience 
            rather than any sort of design requirement
            """
            pos_seg[_ann_idx].append((_anc_idx, _overlap, _len_offset, _cen_offset, _sentence_idx))
            anc_len_to_n_pos[anc_len] += 1

            anc_to_ann[anc_idx] = (_overlap, _ann_idx)

            total_n_pos += 1
            break

        if not filled and len(potential_matches) > 0:
            """
            if all the matching GT segments for this anchor are already in the list of 
            positive GT segments (i.e. they also matched earlier anchors),
            choose a single random segment from the matches and add it there as well;
            seems to be a way to ensure that at least one and only one matching GT segment gets added for each anchor;
            the problem of duplication is somewhat ameliorated by the fact that this one will have 
            different offsets
            """
            shuffle(potential_matches)
            item = potential_matches[0]
            pos_seg[item[0]].append(tuple(item[1:]))

    out_txt = ''
    # print()
    for anc_len, n_pos in anc_len_to_n_pos.items():
        out_txt += f'anc_len: {anc_len} n_pos: {n_pos} ({float(n_pos) / total_n_pos * 100}%)\n'
    # print()

    # print()
    for gt_len, count in gt_len_to_count.items():
        out_txt += f'gt_len: {gt_len} count: {count} ({float(count) / len(annotations) * 100}%)\n'
    # print()

    n_miss_props = 0
    matched_ann_idx = list(pos_seg.keys())
    all_ann_ids = list(range(n_annotations))
    n_matched_ann = len(matched_ann_idx)

    unmatched_ann_idx = set(all_ann_ids) - set(matched_ann_idx)

    if matched_ann_idx != n_annotations:
        n_miss_props = n_annotations - n_matched_ann
        pc_miss_props = (n_miss_props / n_annotations) * 100
        out_txt += f'{n_miss_props} / {n_annotations} ({pc_miss_props}%) annotations in {vid} have no matching ' \
            f'proposal\n'

        # print()
        for ann_idx in unmatched_ann_idx:
            unmatched_ann = annotations[ann_idx]
            gt_s, gt_e = unmatched_ann['segment']
            seg_id = unmatched_ann['id']

            overlap, anc_idx = ann_overlap[ann_idx]
            anc_cen = anc_cen_all[anc_idx]
            anc_len = anc_len_all[anc_idx]

            txt = f'seg {seg_id}: {gt_s / sampling_sec:.2f} - {gt_e / sampling_sec:.2f}  ' \
                f'ov: {overlap:.3f} ' \
                f'anc cen: {anc_cen} len: {anc_len}'

            if overlap > pos_thresh:
                max_overlap, max_ann_idx = anc_to_ann[anc_idx]

                assert max_overlap >= overlap, "max_overlap must exceed overlap"

                max_ann = annotations[max_ann_idx]
                max_gt_s, max_gt_e = max_ann['segment']
                max_seg_id = max_ann['id']

                txt += f' max_seg {max_seg_id}: {max_gt_s / sampling_sec:.2f} - {max_gt_e / sampling_sec:.2f} ' \
                    f'max_ov: {max_overlap:.3f} '

            out_txt += txt + '\n'
            # print(txt)
        # print()

    """list of anchors having max overlap with any GT segment < neg_thresh"""
    neg_seg = []
    for oi, overlap in enumerate(neg_overlap):
        if isinstance(overlap, np.ndarray):
            overlap = float(overlap.item())

        if overlap < neg_thresh:
            neg_seg.append((oi, overlap))

    npos_seg = 0
    for k in pos_seg:
        npos_seg += len(pos_seg[k])

    out_txt += f'pos anc: {npos_seg}, neg anc: {len(neg_seg)}\n'

    n_pos_seg = 0
    sample_list = []
    for k in pos_seg:
        """
        all neg_segs in each sequence are the same, since they need to be negative
        for all samples
        """
        neg_anc_info = neg_seg

        """
        all_segs is a list of 5-tuples: anc_idx, overlap, len_offset, cen_offset, ann['sentence_idx']
        however, each such list has only a single such tuple
        """
        all_segs = pos_seg[k]

        """sentence_idx - same for all entries in all_segs"""
        sentence_idx = all_segs[0][-1]  # [s[-1] for s in all_segs]

        """anc_idx, overlap, len_offset, cen_offset"""
        pos_anc_info = [s[:-1] for s in all_segs]

        sample_list.append(
            (video_prefix, vid_frame_ids, pos_anc_info, sentence_idx, neg_anc_info, n_frames))
        n_pos_seg += len(pos_seg[k])

    if save_samplelist:
        sample_list_size = sys.getsizeof(sample_list)
        sample_list_size_mb = sample_list_size / 1e6
        sample_list_path = os.path.join(sample_list_dir, f'{vid}.pkl')
        print(f'\n{vid} : saving sample_list of size {sample_list_size_mb} MB to {sample_list_path}')
        with open(sample_list_path, 'wb') as f:
            pickle.dump(sample_list, f)
        print(f'\n{vid} : done')
    out_txt_path = os.path.join(out_txt_dir, f'{vid}.log')

    print(f'\n{vid} : saving out_txt to {out_txt_path}')
    with open(out_txt_path, 'w') as fid:
        fid.write(out_txt)
    print(f'\n{vid} : done')

    return sample_list, video_prefix, vid_frame_ids, n_frames, pos_seg, neg_seg, n_miss_props, n_pos_seg
