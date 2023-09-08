"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# general packages
import os
import sys
import numpy as np
import random
import time

from tqdm import tqdm

# import paramparse

# # torch
import torch

# torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F
# from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import torch.utils.data.distributed

from train_params import TrainParams, get_args

# misc
from dnc_data.anet_dataset import ANetDataset, anet_collate_fn, get_vocab_and_sentences
from model.action_prop_dense_cap import ActionPropDenseCap, DropoutTime1D
from model.transformer import Attention, MultiHead, LayerNorm, ResidualBlock, FeedForward, \
    EncoderLayer, Encoder, Transformer, DecoderLayer, Decoder, RealTransformer
from dnc_utilities import get_latest_checkpoint, excel_ids_to_grid, diff_sentence_to_grid_cells, FeatureExtractor

import dnc_to_mot

home_path = os.path.expanduser('~')

swi_path = os.path.join(home_path, 'ipsc', 'ipsc_static_segmentation', 'swin_detection')
sys.path.append(swi_path)

import mmcv
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.models import build_detector
from mmdet.utils.misc import read_class_info, linux_path

dmdp_path = os.path.join(home_path, 'isl_labeling_tool', 'deep_mdp')
sys.path.append(dmdp_path)

from input import Input
from objects import Annotations

from utilities import linux_path, CustomLogger


def get_dataset(sampling_sec, feat_model, params: TrainParams):
    # process text
    train_val_splits = [params.train_splits[0], params.val_splits[0]]
    sample_list_dir = os.path.dirname(params.train_samplelist_path)
    text_proc, raw_data, n_videos = get_vocab_and_sentences(
        params.dataset_file,
        train_val_splits,
        # args.max_sentence_len,
        save_path=sample_list_dir)

    # Create the dataset and data loader instance
    train_dataset = ANetDataset(
        feat_model=feat_model,
        feat_shape=params.feat_shape,
        image_path=params.feature_root,
        n_vids=n_videos['training'],
        splits=params.train_splits,
        slide_window_size=params.slide_window_size,
        dur_file=params.dur_file,
        kernel_list=params.kernel_list,
        text_proc=text_proc,
        raw_data=raw_data,
        pos_thresh=params.pos_thresh,
        neg_thresh=params.neg_thresh,
        stride_factor=params.stride_factor,
        enable_flow=params.enable_flow,
        dataset=params.dataset,
        sampling_sec=sampling_sec,
        save_samplelist=params.save_train_samplelist,
        load_samplelist=params.load_train_samplelist,
        sample_list_dir=params.train_samplelist_path,
    )

    valid_dataset = ANetDataset(
        feat_model=feat_model,
        feat_shape=params.feat_shape,
        image_path=params.feature_root,
        n_vids=n_videos['validation'],
        splits=params.val_splits,
        slide_window_size=params.slide_window_size,
        dur_file=params.dur_file,
        kernel_list=params.kernel_list,
        text_proc=text_proc,
        raw_data=raw_data,
        pos_thresh=params.pos_thresh,
        neg_thresh=params.neg_thresh,
        stride_factor=params.stride_factor,
        enable_flow=params.enable_flow,
        dataset=params.dataset,
        sampling_sec=sampling_sec,
        save_samplelist=params.save_valid_samplelist,
        load_samplelist=params.load_valid_samplelist,
        sample_list_dir=params.valid_samplelist_path,
    )

    # if text_proc is not None:
    #     exit()

    if params.distributed and params.cuda:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if not train_dataset.samples_loaded:
        train_dataset.get_samples(params.n_proc)
    if not valid_dataset.samples_loaded:
        valid_dataset.get_samples(params.n_proc)

    return train_dataset, valid_dataset, text_proc, train_sampler


def get_model(text_proc, feat_model, args):
    """

    :param text_proc:
    :param TrainParams args:
    :return:
    """
    sent_vocab = text_proc.vocab
    max_sentence_len = text_proc.fix_length
    model = ActionPropDenseCap(
        feat_model=feat_model,
        feat_shape=args.feat_shape,
        enable_flow=args.enable_flow,
        rgb_ch=args.rgb_ch,
        dim_model=args.d_model,
        dim_hidden=args.d_hidden,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab=sent_vocab,
        in_emb_dropout=args.in_emb_dropout,
        attn_dropout=args.attn_dropout,
        vis_emb_dropout=args.vis_emb_dropout,
        cap_dropout=args.cap_dropout,
        nsamples=args.train_sample,
        kernel_list=args.kernel_list,
        stride_factor=args.stride_factor,
        learn_mask=args.mask_weight > 0,
        max_sentence_len=max_sentence_len,
        window_length=args.slide_window_size,
    )

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from,
                                         map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        model.cuda()
        if args.distributed:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True,
            )
        elif torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model.cuda()
    return model


def get_feat_extractor(feat_cfg, ckpt, fuse_conv_bn):
    cfg_dict = Config.fromfile(feat_cfg)
    # if params.cfg_options is not None:
    #     cfg_dict.merge_from_dict(params.cfg_options)
    # import modules from string list.
    if cfg_dict.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg_dict['custom_imports'])
    # set cudnn_benchmark
    if cfg_dict.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg_dict.model.pretrained = None
    if cfg_dict.model.get('neck'):
        if isinstance(cfg_dict.model.neck, list):
            for neck_cfg in cfg_dict.model.neck:
                if neck_cfg.get('rfp_backbone'):
                    if neck_cfg.rfp_backbone.get('pretrained'):
                        neck_cfg.rfp_backbone.pretrained = None
        elif cfg_dict.model.neck.get('rfp_backbone'):
            if cfg_dict.model.neck.rfp_backbone.get('pretrained'):
                cfg_dict.model.neck.rfp_backbone.pretrained = None

    # build the model and load checkpoint
    cfg_dict.model.train_cfg = None
    model = build_detector(cfg_dict.model, test_cfg=cfg_dict.get('test_cfg'))
    fp16_cfg = cfg_dict.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, ckpt, map_location='cpu')
    if fuse_conv_bn:
        model = fuse_conv_bn(model)
    return model


def main():
    params = get_args()  # type: TrainParams

    print(f'params.resume: {params.resume}')

    sampled_frames = params.sampled_frames
    sampling_sec = float(sampled_frames) / float(params.fps)

    feat_model = None

    if params.feat_cfg:
        feat_cfg_name = os.path.splitext(os.path.basename(params.feat_cfg))[0]

        print(f'loading feature extractor from {feat_cfg_name}')

        params.feat_cfg = linux_path(swi_path, params.feat_cfg)

        if not params.feat_ckpt:
            params.feat_ckpt = 'latest.pth'

        ckpt_dir = linux_path(swi_path, 'work_dirs', feat_cfg_name)
        params.feat_ckpt = linux_path(ckpt_dir, params.feat_ckpt)

        feat_model = get_feat_extractor(params.feat_cfg, params.feat_ckpt, params.fuse_conv_bn)

        if params.cuda:
            feat_model = feat_model.cuda()

        feat_model = FeatureExtractor(
            feat_model=feat_model,
            reduction=params.feat_reduction,
            batch_size=params.feat_batch_size,
            norm=(params.mean, params.std),
            cuda=params.cuda,
        )

    # dist parallel, optional
    # params.distributed = params.world_size > 1
    # params.distributed = 1

    if params.distributed and params.cuda:
        torch.cuda.set_device(params.local_rank)

        from datetime import timedelta

        from urllib.parse import urlparse

        params.dist_url = f'{params.dist_url}-{int(time.time_ns())}'

        k = urlparse(params.dist_url)
        print()
        print(f'dist_url scheme: {k.scheme}')
        print(f'dist_url path: {k.path}')
        print()

        if k.scheme == 'file' and os.path.exists(k.path):
            print(f'removing existing dist_url path: {k.path}')
            os.remove(k.path)
        dist.init_process_group(
            backend='nccl',
            init_method="env://",
            # backend=params.dist_backend,
            # init_method=params.dist_url,
            # world_size=params.world_size,
            # rank=params.local_rank,
            # timeout=timedelta(seconds=100),
        )

    # params = TrainParams()
    # paramparse.process(params)

    # arguments inspection

    print(f'params.train_splits: {params.train_splits}')
    print(f'params.train_splits[0]: {params.train_splits[0]}')

    if params.valid_batch_size <= 0:
        params.valid_batch_size = params.batch_size

    # print(f'params.val_splits: {params.val_splits}')
    # print(f'params.val_splits[0]: {params.val_splits[0]}')

    """
    slide_window_size is in units of SAMPLED frames rather than original ones
    this is also a misnomer since there is an implicit and 
    mind bogglingly annoying assumption underlying this entire gunky operation that none of the 
    training or testing videos exceed this length otherwise the excess part will be ignored rather than 
    any kind of actual sliding window operation happening to process the long video piecewise
    """
    assert (params.slide_window_size >= params.slide_window_stride)
    # assert (params.sampling_sec == 0.5)  # attention! sampling_sec is hard coded as 0.5

    if not params.train_samplelist_path:
        params.train_samplelist_path = linux_path(params.ckpt, f"{params.train_splits[0]}_samples")
        # print(f'params.train_samplelist_path: {params.train_samplelist_path}')

    # if not params.train_sentence_dict_path:
    #     params.train_sentence_dict_path = linux_path(params.ckpt, "train_sentence_dict.pkl")

    if not params.valid_samplelist_path:
        params.valid_samplelist_path = linux_path(params.ckpt, f"{params.val_splits[0]}_samples")
        # print(f'params.valid_samplelist_path: {params.valid_samplelist_path}')

    # exit()

    # if not params.valid_sentence_dict_path:
    #     params.valid_sentence_dict_path = linux_path(params.ckpt, "valid_sentence_dict.pkl")

    print(f'save_valid_samplelist: {params.save_valid_samplelist}')
    print(f'save_train_samplelist: {params.save_train_samplelist}')
    print(f'valid_samplelist_path: {params.valid_samplelist_path}')
    print(f'train_samplelist_path: {params.train_samplelist_path}')

    if params.db_root:
        params.feature_root = linux_path(params.db_root, params.feature_root)
        params.dataset_file = linux_path(params.db_root, params.dataset_file)
        params.dur_file = linux_path(params.db_root, params.dur_file)

    print('loading dataset')
    train_dataset, valid_dataset, text_proc, train_sampler = get_dataset(sampling_sec, feat_model, params)

    assert tuple(train_dataset.feat_shape) == tuple(params.feat_shape), "train_dataset feat_shape mismatch"
    assert tuple(valid_dataset.feat_shape) == tuple(params.feat_shape), "valid_dataset feat_shape mismatch"

    print('building model')
    model = get_model(text_proc, feat_model, params)

    print('initializing weights')

    def weights_init(m):
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, (
                DropoutTime1D,
                torch.nn.ReLU,
                torch.nn.Flatten,
                torch.nn.Dropout,
                torch.nn.BCEWithLogitsLoss,
                torch.nn.SmoothL1Loss,
                torch.nn.MSELoss,
        )):
            pass
        elif isinstance(m, (
                ActionPropDenseCap,
                Attention,
                MultiHead,
                LayerNorm,
                ResidualBlock,
                FeedForward,
                EncoderLayer,
                Encoder,
                Transformer,
                DecoderLayer,
                Decoder,
                RealTransformer,
        )):
            for _m in m.children():
                weights_init(_m)
        elif isinstance(m, (
                torch.nn.Sequential,
                torch.nn.ModuleList,)):
            for _m in m:
                weights_init(_m)
        else:
            raise AssertionError('unknown layer type')

    try:
        module = model.module
    except AttributeError:
        module = model

    module.apply(weights_init)

    # for _p in model.parameters():
    #     print(_p)

    train_loader = DataLoader(train_dataset,
                              batch_size=params.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=params.num_workers,
                              collate_fn=anet_collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=params.valid_batch_size,
                              shuffle=False,
                              num_workers=params.num_workers,
                              collate_fn=anet_collate_fn)

    torch.manual_seed(int(time.time()))
    np.random.seed(int(time.time()))
    random.seed(int(time.time()))

    if params.cuda:
        torch.cuda.manual_seed_all(int(time.time()))

    os.makedirs(params.ckpt, exist_ok=True)

    # filter params that don't require gradient (credit: PyTorch Forum issue 679)
    # smaller learning rate for the decoder
    if params.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            params.learning_rate, betas=(params.alpha, params.beta), eps=params.epsilon)
    elif params.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            params.learning_rate,
            weight_decay=1e-5,
            momentum=params.alpha,
            nesterov=True
        )
    else:
        raise NotImplementedError

    # learning rate decay every 1 epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=params.reduce_factor,
                                               patience=params.patience_epoch,
                                               verbose=True)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.6)

    # Number of parameter blocks in the network
    print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    best_train_loss = float('inf')
    best_train_loss_epoch = None

    best_valid_loss = float('inf')
    best_valid_loss_epoch = None

    all_eval_losses = []
    all_cls_losses = []
    all_reg_losses = []
    all_sent_losses = []
    all_mask_losses = []
    all_training_losses = []

    tb_path = linux_path(params.ckpt, 'tb')
    vis_path = linux_path(params.ckpt, 'vis')
    os.makedirs(tb_path, exist_ok=1)
    os.makedirs(vis_path, exist_ok=1)

    print(f'saving tensorboard log to: {tb_path}')

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tb_path)

    if params.vocab_fmt == 0:
        word_to_grid_cell = excel_ids_to_grid(params.grid_res)
        sentence_to_grid_cells = lambda words: [word_to_grid_cell[word] for word in words]
    else:
        import functools
        sentence_to_grid_cells = functools.partial(diff_sentence_to_grid_cells,
                                                   fmt_type=params.vocab_fmt,
                                                   max_diff=params.max_diff,
                                                   )

    start_epoch = 0

    if params.resume:
        checkpoint, ckpt_epoch = get_latest_checkpoint(params.ckpt, ignore_missing=True)
        if checkpoint is not None:
            print(f"loading weights from {checkpoint}")
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict)
            start_epoch = ckpt_epoch + 1

    for train_epoch in range(start_epoch, params.max_epochs):
        t_epoch_start = time.time()
        # print('Epoch: {}'.format(train_epoch))

        if params.distributed:
            train_sampler.set_epoch(train_epoch)

        train_loss_dict = train(
            train_epoch,
            model,
            optimizer,
            train_loader,
            vis_path,
            sampled_frames,
            sampling_sec,
            sentence_to_grid_cells,
            params)

        epoch_loss = np.mean(train_loss_dict['loss'])
        cls_loss = np.mean(train_loss_dict['cls_loss'])
        reg_loss = np.mean(train_loss_dict['reg_loss'])
        sent_loss = np.mean(train_loss_dict['sent_loss'])

        writer.add_scalar('train/loss', epoch_loss, train_epoch)
        writer.add_scalar('train/cls_loss', cls_loss, train_epoch)
        writer.add_scalar('train/reg_loss', reg_loss, train_epoch)
        writer.add_scalar('train/sent_loss', sent_loss, train_epoch)

        if train_loss_dict['mask_loss']:
            mask_loss = np.mean(train_loss_dict['mask_loss'])
            writer.add_scalar('train/mask_loss', mask_loss, train_epoch)

        if train_loss_dict['scst_loss']:
            scst_loss = np.mean(train_loss_dict['scst_loss'])
            writer.add_scalar('train/scst_loss', scst_loss, train_epoch)

        all_training_losses.append(epoch_loss)

        if train_epoch % params.save_checkpoint_every == 0 or train_epoch == params.max_epochs:
            if (params.distributed and dist.get_rank() == 0) or not params.distributed:
                del_epoch = train_epoch - params.keep_checkpoints
                del_ckpt = os.path.join(params.ckpt, f'epoch_{del_epoch}.pth')
                if del_epoch >= 0 and os.path.exists(del_ckpt):
                    os.remove(del_ckpt)

                ckpt = os.path.join(params.ckpt, f'epoch_{train_epoch}.pth')
                print(f'saving regular checkpoint: {ckpt}')

                torch.save(model.state_dict(), ckpt)

        if train_epoch % params.validate_every == 0 or train_epoch == params.max_epochs:

            (valid_loss, val_cls_loss,
             val_reg_loss, val_sent_loss, val_mask_loss) = validate(
                train_epoch,
                model,
                valid_loader,
                vis_path,
                sampled_frames,
                sampling_sec,
                sentence_to_grid_cells,
                params)

            writer.add_scalar('val/loss', valid_loss, train_epoch)
            writer.add_scalar('val/cls_loss', val_cls_loss, train_epoch)
            writer.add_scalar('val/reg_loss', val_reg_loss, train_epoch)
            writer.add_scalar('val/sent_loss', val_sent_loss, train_epoch)
            writer.add_scalar('val/mask_loss', val_mask_loss, train_epoch)

            all_eval_losses.append(valid_loss)
            all_cls_losses.append(val_cls_loss)
            all_reg_losses.append(val_reg_loss)
            all_sent_losses.append(val_sent_loss)
            all_mask_losses.append(val_mask_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_loss_epoch = train_epoch
                if (params.distributed and dist.get_rank() == 0) or not params.distributed:
                    checkpoint, epoch = get_latest_checkpoint(params.ckpt, 'best_val_model_', True)
                    if checkpoint is not None:
                        os.remove(checkpoint)
                    ckpt_path = os.path.join(params.ckpt, f'best_val_model_{train_epoch}.pth')
                    torch.save(model.state_dict(), ckpt_path)
                print('*' * 5)
                print('Better validation loss {:.4f} found, save model'.format(valid_loss))
            # learning rate decay
            scheduler.step(valid_loss)

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            best_train_loss_epoch = train_epoch
            if (params.distributed and dist.get_rank() == 0) or not params.distributed:
                checkpoint, epoch = get_latest_checkpoint(params.ckpt, 'best_train_model_', True)
                if checkpoint is not None:
                    os.remove(checkpoint)
                ckpt_path = os.path.join(params.ckpt, f'best_train_model_{train_epoch}.pth')
                torch.save(model.state_dict(), ckpt_path)
            print('*' * 5)
            print(f'Better training loss {epoch_loss:.4f} found, save model')


def train(
        epoch,
        model,
        optimizer,
        train_loader,
        vis_path,
        sampled_frames,
        sampling_sec,
        sentence_to_grid_cells,
        params: TrainParams):
    model.train()  # training mode

    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model

    train_loss = []
    train_cls_loss = []
    train_reg_loss = []
    train_sent_loss = []
    train_scst_loss = []
    train_mask_loss = []

    nbatches = len(train_loader)
    # t_iter_start = time.time()

    pbar = tqdm(train_loader, total=nbatches, ncols=140, desc=f'training epoch {epoch}')

    if epoch >= params.vis_from:
        vis_batch_id = random.randint(0, nbatches - 1)
        print(f'\nvisualizing batch {vis_batch_id}\n')

    sample_prob = min(params.sample_prob, int(epoch / 5) * 0.05)

    inference_t = vis_t = 0

    for train_iter, data in enumerate(pbar):
        global_iter = epoch * nbatches + train_iter

        img_batch, frame_length, video_prefix_list, feat_frame_ids_list, samples, times = data
        tempo_seg_pos, tempo_seg_neg, sentence_batch = samples

        tempo_seg_pos_ = tempo_seg_pos.cpu().numpy()
        tempo_seg_neg_ = tempo_seg_neg.cpu().numpy()
        sentence_batch_ = sentence_batch.cpu().numpy()

        load_t, torch_t, collate_t = times
        if params.cuda:
            img_batch = img_batch.cuda()
            tempo_seg_neg = tempo_seg_neg.cuda()
            tempo_seg_pos = tempo_seg_pos.cuda()
            sentence_batch = sentence_batch.cuda()

        start_t = time.time()
        result = model(
            img_batch,
            tempo_seg_pos,
            tempo_seg_neg,
            sentence_batch,
            sample_prob,
            params.stride_factor,
            scst=params.scst_weight > 0,
            gated_mask=params.gated_mask
        )

        pred_score, gt_score, pred_offsets, gt_offsets, pred_sentence, gt_sent, scst_loss, mask_loss = result

        cls_loss = module.bce_loss(pred_score, gt_score) * params.cls_weight
        reg_loss = module.reg_loss(pred_offsets, gt_offsets) * params.reg_weight
        sent_loss = F.cross_entropy(pred_sentence, gt_sent) * params.sent_weight
        total_loss = cls_loss + reg_loss + sent_loss

        if scst_loss is not None:
            scst_loss *= params.scst_weight
            total_loss += scst_loss

            train_scst_loss.append(scst_loss.data.item())
            # writer.add_scalar('train_iter/scst_loss', train_scst_loss[-1], global_iter)

        if mask_loss is not None:
            mask_loss = params.mask_weight * mask_loss
            total_loss += mask_loss

            train_mask_loss.append(mask_loss.data.item())
            # writer.add_scalar('train_iter/mask_loss', train_mask_loss[-1], global_iter)

        else:
            mask_loss = cls_loss.new(1).fill_(0)

        train_cls_loss.append(cls_loss.data.item())
        train_reg_loss.append(reg_loss.data.item())
        train_sent_loss.append(sent_loss.data.item())
        train_loss.append(total_loss.data.item())

        end_t = time.time()
        forward_t = (end_t - start_t) * 1000

        if epoch >= params.vis_from and train_iter == vis_batch_id:
            batch_size = img_batch.size()[0]
            vis_sample_id = random.randint(0, batch_size - 1)
            print(f'\nvisualizing sample {vis_sample_id}\n')

            img_batch_vis = img_batch[vis_sample_id:vis_sample_id + 1, ...]
            video_prefix = video_prefix_list[vis_sample_id]
            feat_frame_ids = feat_frame_ids_list[vis_sample_id]

            model.eval()
            inference_t, vis_t = visualize(
                epoch=epoch,
                module=module,
                img_batch_vis=img_batch_vis,
                video_prefix=video_prefix,
                feat_frame_ids=feat_frame_ids,
                sampled_frames=sampled_frames,
                frame_length=frame_length,
                sampling_sec=sampling_sec,
                vis_path=vis_path,
                sentence_to_grid_cells=sentence_to_grid_cells,
                params=params)

            model.train()

        optimizer.zero_grad()
        total_loss.backward()

        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                          params.grad_norm)

        optimizer.step()

        # pbar.set_description(f'training epoch {epoch} '
        #                      f'(data: {load_t:.2f}, {torch_t:.2f}, {collate_t:.2f}) '
        #                      f'(model: {forward_t:.2f},{inference_t:.2f},{vis_t:.2f})'
        #                      )

    loss_dict = {
        'loss': train_loss,
        'cls_loss': train_cls_loss,
        'reg_loss': train_reg_loss,
        'sent_loss': train_sent_loss,
        'mask_loss': train_mask_loss,
        'scst_loss': train_scst_loss,
    }
    return loss_dict


def validate(epoch,
             model,
             loader,
             vis_path,
             sampled_frames,
             sampling_sec,
             sentence_to_grid_cells,
             params: TrainParams):
    if hasattr(model, 'module'):
        module = model.module
    else:
        module = model

    model.eval()
    valid_loss = []
    val_cls_loss = []
    val_reg_loss = []
    val_sent_loss = []
    val_mask_loss = []

    nbatches = len(loader)
    pbar = tqdm(loader, total=nbatches, ncols=140, desc=f'validation epoch {epoch}')

    if epoch >= params.vis_from:
        vis_batch_id = random.randint(0, nbatches - 1)
        print(f'\nvisualizing batch {vis_batch_id}\n')

    for val_iter, data in enumerate(pbar):
        inference_t = vis_t = 0

        global_iter = epoch * nbatches + val_iter

        img_batch, frame_length, video_prefix_list, feat_frame_ids_list, samples, times = data
        tempo_seg_pos, tempo_seg_neg, sentence_batch = samples
        load_t, torch_t, collate_t = times

        # img_batch = Variable(img_batch)
        # tempo_seg_pos = Variable(tempo_seg_pos)
        # tempo_seg_neg = Variable(tempo_seg_neg)
        # sentence_batch = Variable(sentence_batch)

        if params.cuda:
            img_batch = img_batch.cuda()
            tempo_seg_neg = tempo_seg_neg.cuda()
            tempo_seg_pos = tempo_seg_pos.cuda()
            sentence_batch = sentence_batch.cuda()

        with torch.no_grad():
            start_t = time.time()
            (pred_score, gt_score,
             pred_offsets, gt_offsets,
             pred_sentence, gt_sent,
             _, mask_loss) = model(img_batch, tempo_seg_pos,
                                   tempo_seg_neg, sentence_batch,
                                   stride_factor=params.stride_factor,
                                   gated_mask=params.gated_mask)

            cls_loss = module.bce_loss(pred_score, gt_score) * params.cls_weight
            reg_loss = module.reg_loss(pred_offsets, gt_offsets) * params.reg_weight
            sent_loss = F.cross_entropy(pred_sentence, gt_sent) * params.sent_weight

            total_loss = cls_loss + reg_loss + sent_loss

            if mask_loss is not None:
                mask_loss = params.mask_weight * mask_loss
                total_loss += mask_loss
            else:
                mask_loss = cls_loss.new(1).fill_(0)

            val_cls_loss.append(cls_loss.data.item())
            val_reg_loss.append(reg_loss.data.item())
            val_sent_loss.append(sent_loss.data.item())
            val_mask_loss.append(mask_loss.data.item())
            valid_loss.append(total_loss.data.item())

        end_t = time.time()
        forward_t = (end_t - start_t) * 1000

        if epoch >= params.vis_from and val_iter == vis_batch_id:
            batch_size = img_batch.size()[0]
            vis_sample_id = random.randint(0, batch_size - 1)
            print(f'\nvisualizing sample {vis_sample_id}\n')

            img_batch_vis = img_batch[vis_sample_id:vis_sample_id + 1, ...]
            video_prefix = video_prefix_list[vis_sample_id]
            feat_frame_ids = feat_frame_ids_list[vis_sample_id]

            inference_t, vis_t = visualize(
                epoch=epoch,
                module=module,
                img_batch_vis=img_batch_vis,
                video_prefix=video_prefix,
                feat_frame_ids=feat_frame_ids,
                sampled_frames=sampled_frames,
                frame_length=frame_length,
                sampling_sec=sampling_sec,
                vis_path=vis_path,
                sentence_to_grid_cells=sentence_to_grid_cells,
                params=params,
            )

        # pbar.set_description(f'validation epoch {epoch} '
        #                      f'(data: {load_t:.2f},{torch_t:.2f},{collate_t:.2f}) '
        #                      f'(model: {forward_t:.2f},{inference_t:.2f},{vis_t:.2f})'
        #                      )

    return (np.mean(valid_loss), np.mean(val_cls_loss),
            np.mean(val_reg_loss), np.mean(val_sent_loss), np.mean(val_mask_loss))


def visualize(
        epoch,
        module: ActionPropDenseCap,
        img_batch_vis,
        video_prefix,
        feat_frame_ids,
        sampled_frames,
        frame_length,
        sampling_sec,
        vis_path,
        sentence_to_grid_cells,
        params: TrainParams):
    invalid_words = ['<UNK>', ]
    start_t = time.time()

    with torch.no_grad():
        all_proposal_results = module.inference(
            x=img_batch_vis,
            actual_frame_length=frame_length,
            sampling_sec=sampling_sec,
            min_prop_num=params.min_prop_num,
            max_prop_num=params.max_prop_num,
            min_prop_num_before_nms=params.min_prop_before_nms,
            pos_thresh=params.pos_thresh,
            stride_factor=params.stride_factor,
            gated_mask=params.gated_mask)

    end_t = time.time()
    inference_t = (end_t - start_t) * 1000

    _input = None
    start_t = time.time()

    annotations = []

    assert len(all_proposal_results) == 1, "annoying invalid all_proposal_results"

    for pred_start, pred_end, pred_s, sentence in all_proposal_results[0]:
        traj_n_frames = pred_end - pred_start

        if traj_n_frames <= 2:
            # print(f'skipping trajextory with too few frames')
            continue

        words = sentence.upper().split(' ')

        words = [word for word in words if word not in invalid_words]

        if len(words) < 2:
            continue

        pred_start_t = pred_start * sampling_sec
        pred_end_t = pred_end * sampling_sec

        # pred_start_frame = pred_start_t * args.fps
        # pred_end_frame = pred_end_t * args.fps

        sentence = ' '.join(words)

        annotations.append(
            {
                'sentence': sentence,
                'segment': [pred_start_t, pred_end_t]
            })

    if not annotations:
        print('\nno valid annotations found for visualization\n')
        return inference_t, 0

    n_traj = len(annotations)

    if n_traj > params.max_vis_traj:
        random.shuffle(annotations)
        annotations = annotations[:params.max_vis_traj]

    start_id = end_id = -1

    vid = os.path.basename(video_prefix)

    if '--' in vid:
        vid_name, vid_frame_ids = vid.split('--')
        vid_frame_ids = tuple(map(int, vid_frame_ids.split('_')))
        start_id, end_id = vid_frame_ids
    else:
        vid_name = vid

        if feat_frame_ids is not None:
            feat_start_id, feat_end_id = feat_frame_ids
            start_id, end_id = int(feat_start_id * sampled_frames), int(feat_end_id * sampled_frames)

    out_name = f'{epoch}-{vid_name}'
    if _input is None:
        src_dir_path = params.db_root
        if params.img_dir_name:
            src_dir_path = linux_path(src_dir_path, params.img_dir_name)
        vid_path = linux_path(src_dir_path, vid_name)

        _input_params = Input.Params(source_type=-1,
                                     batch_mode=True,
                                     path=vid_path,
                                     frame_ids=(start_id, end_id - 1))

        _logger = CustomLogger.setup(__name__)
        _input = Input(_input_params, _logger)

        if not _input.initialize(None):
            _logger.error('Input pipeline could not be initialized')
            return False

    dnc_to_mot.run(
        dnc_data=annotations,
        frames=_input.all_frames,
        seq_info=None,
        json_data=None,
        n_seq=None,
        out_dir=vis_path,
        out_name=out_name,
        sentence_to_grid_cells=sentence_to_grid_cells,
        grid_res=params.grid_res,
        fps=params.fps,
        vis=params.vis,
        params=None,
    )
    end_t = time.time()

    vis_t = (end_t - start_t) * 1000

    return inference_t, vis_t


if __name__ == "__main__":
    main()
