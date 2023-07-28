"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import sys

sys.path.append('../isl_labeling_tool/deep_mdp')

import os
from collections import defaultdict
from tqdm import tqdm
import json
import yaml

import torch
from torch.utils.data import DataLoader
from densecap_utilities import get_latest_checkpoint

import paramparse

from test_params import TestParams
from densecap_data.anet_test_dataset import ANetTestDataset, anet_test_collate_fn
from densecap_data.anet_dataset import get_vocab_and_sentences
from densecap_data.utils import update_values

from model.action_prop_dense_cap import ActionPropDenseCap
from tools.eval_proposal_anet import ANETproposal

from utilities import linux_path


def get_dataset(args):
    """

    :param TestParams args:
    :return:
    """

    assert args.sample_list_path, "sample_list_path must be provided"
    # process text
    test_split = [args.test_split, ]
    text_proc, raw_data, n_videos = get_vocab_and_sentences(
        args.dataset_file,
        test_split,
        save_path=args.sample_list_path)

    # Create the dataset and data loader instance
    test_dataset = ANetTestDataset(args.feature_root,
                                   args.slide_window_size,
                                   args.dur_file,
                                   args.dataset,
                                   args.sampling_sec,
                                   text_proc,
                                   raw_data,
                                   args.test_split,
                                   args.learn_mask,
                                   args.sample_list_path)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=anet_test_collate_fn)

    return test_loader, test_dataset, text_proc


def get_model(text_proc, args):
    """

    :param text_proc:
    :param TestParams args:
    :return:
    """
    sent_vocab = text_proc.vocab
    model = ActionPropDenseCap(dim_model=args.d_model,
                               dim_hidden=args.d_hidden,
                               n_layers=args.n_layers,
                               n_heads=args.n_heads,
                               vocab=sent_vocab,
                               in_emb_dropout=args.in_emb_dropout,
                               attn_dropout=args.attn_dropout,
                               vis_emb_dropout=args.vis_emb_dropout,
                               cap_dropout=args.cap_dropout,
                               nsamples=0,
                               kernel_list=args.kernel_list,
                               stride_factor=args.stride_factor,
                               learn_mask=args.learn_mask)

    model = torch.nn.DataParallel(model)

    if args.cuda:
        model.cuda()

    print(f"loading weights from {args.ckpt}")
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict)

    # if len(args.ckpt) > 0:
    #     print("Initializing weights from {}".format(args.ckpt))
    #     model.load_state_dict(torch.load(args.ckpt,
    #                                      map_location=lambda storage, location: storage))

    return model


def validate(model, loader, dataset, out_dir, args):
    """

    :param model:
    :param loader:
    :param TestParams args:
    :return:
    """
    model.eval()
    densecap_result = defaultdict(list)
    prop_result = defaultdict(list)

    avg_prop_num = 0
    nbatches = len(loader)
    pbar = tqdm(loader, total=nbatches, ncols=120)

    for _iter, data in enumerate(pbar):
        image_feat, original_num_frame, video_prefix, times = data
        load_t, torch_t, collate_t = times

        pbar.set_description(f'times: {load_t:.2f}, {torch_t:.2f}, {collate_t:.2f}')

        video_name = os.path.basename(video_prefix[0])

        with torch.no_grad():
            # image_feat = Variable(image_feat)
            # ship data to gpu
            if args.cuda:
                image_feat = image_feat.cuda()

            # dtype = image_feat.data.type()
            # if video_name not in dataset.frame_to_second:
            #     dataset.frame_to_second[video_name] = args.sampling_sec
            #     print(f"cannot find frame_to_second for video {video_name}")

            sampling_sec = dataset.frame_to_second[video_name]  # batch_size has to be 1
            all_proposal_results = model.module.inference(
                image_feat,
                original_num_frame,
                sampling_sec,
                args.min_prop_num,
                args.max_prop_num,
                args.min_prop_before_nms,
                args.pos_thresh,
                args.stride_factor,
                gated_mask=args.gated_mask)

            for b in range(len(video_prefix)):
                vid = os.path.basename(video_prefix[b])
                for pred_start, pred_end, pred_s, sent in all_proposal_results[b]:
                    pred_start_t = pred_start * sampling_sec
                    pred_end_t = pred_end * sampling_sec

                    pred_start_frame = pred_start_t * args.fps
                    pred_end_frame = pred_end_t * args.fps

                    densecap_result[vid].append(
                        {'sentence': sent,
                         'timestamp': [pred_start_t, pred_end_t]})

                    prop_result[vid].append(
                        {'segment': [pred_start_t, pred_end_t],
                         'score': pred_s})

                avg_prop_num += len(all_proposal_results[b])

        if _iter >= args.max_batches > 0:
            break

    print("average proposal number: {}".format(avg_prop_num / len(loader.dataset)))

    # write captions to json file for evaluation (densecap)
    # dense_cap_all = {
    #     'version': 'VERSION 1.0',
    #     'results': densecap_result,
    #     'external_data': {'used': 'true',
    #                       'details': 'global_pool layer from BN-Inception pretrained from ActivityNet \
    #             and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'}
    # }
    dnc_out_path = linux_path(out_dir, f'densecap.json')
    print(f'dnc_out_path: {dnc_out_path}')
    with open(dnc_out_path, 'w') as f:
        json.dump(densecap_result, f)

    # write proposals to json file for evaluation (proposal)
    prop_out_path = linux_path(out_dir, f'prop.json')
    print(f'prop_out_path: {prop_out_path}')
    # prop_all = {
    #     'version': 'VERSION 1.0',
    #     'results': prop_result,
    #     'external_data': {'used': 'true',
    #                       'details': 'global_pool layer from BN-Inception pretrained from ActivityNet \
    #            and ImageNet (https://github.com/yjxiong/anet2016-cuhk)'}
    # }
    with open(prop_out_path, 'w') as f:
        json.dump(prop_result, f)

    # return eval_results(prop_result, args)


# def eval_results(args):
#     subprocess.Popen(["python2", args.densecap_eval_file, "-s", \
#                       os.path.join('./results/', 'densecap_' + args.val_data_folder + '_' + args.id + '.json'), \
#                       "-v", "-r"] + \
#                      args.densecap_references \
#                      )
#
#     anet_proposal = ANETproposal(args.dataset_file,
#                                  os.path.join('./results/', 'prop_' + args.val_data_folder + '_' + args.id + '.json'),
#                                  tiou_thresholds=np.linspace(0.5, 0.95, 10),
#                                  max_avg_nr_proposals=100,
#                                  subset=args.val_data_folder, verbose=True, check_status=True)
#
#     anet_proposal.evaluate()
#
#     return anet_proposal.area


def main():
    args = TestParams()

    paramparse.process(args)

    with open(args.cfgs_file, 'r') as handle:
        options_yaml = yaml.safe_load(handle)
    update_values(options_yaml, vars(args))

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    assert args.batch_size == 1, "Batch size has to be 1!"

    assert args.slide_window_size > args.slide_window_stride, \
        "slide_window_size must be > slide_window_stride!"

    if args.db_root:
        args.feature_root = linux_path(args.db_root, args.feature_root)
        args.dataset_file = linux_path(args.db_root, args.dataset_file)
        args.dur_file = linux_path(args.db_root, args.dur_file)

    if os.path.isfile(args.ckpt):
        pass
    elif os.path.isdir(args.ckpt):
        if args.ckpt_name:
            ckpt_path = linux_path(args.ckpt, args.ckpt_name)
        else:
            ckpt_path, _ = get_latest_checkpoint(args.ckpt)
        args.ckpt = ckpt_path
    else:
        raise AssertionError(f'invalid ckpt: {args.ckpt}')

    print(f'ckpt: {args.ckpt}')

    ckpt_dir = os.path.dirname(args.ckpt)
    ckpt_name = os.path.splitext(os.path.basename(args.ckpt))[0]

    out_dir = linux_path(ckpt_dir, f'{ckpt_name}_on_{args.test_split}_{args.id}')

    os.makedirs(out_dir, exist_ok=1)

    if not args.sample_list_path:
        args.sample_list_path = linux_path(ckpt_dir, f"{args.test_split}_samples")

    print('loading dataset')
    test_loader, test_dataset, text_proc = get_dataset(args)

    print('building model')
    model = get_model(text_proc, args)

    validate(model, test_loader, test_dataset, out_dir, args)

    # print('proposal recall area: {:.6f}'.format(recall_area))


if __name__ == "__main__":
    main()
