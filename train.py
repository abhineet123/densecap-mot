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

# misc
from densecap_data.anet_dataset import ANetDataset, anet_collate_fn, get_vocab_and_sentences
from model.action_prop_dense_cap import ActionPropDenseCap
from densecap_utilities import get_latest_checkpoint

sys.path.append('../isl_labeling_tool/deep_mdp')

from train_params import TrainParams, get_args
from utilities import linux_path


def get_dataset(args):
    """

    :param TrainParams args:
    :return:
    """
    # process text
    train_val_splits = [args.train_splits[0], args.val_splits[0]]
    sample_list_dir = os.path.dirname(args.train_samplelist_path)
    text_proc, raw_data, n_train_videos, n_val_videos = get_vocab_and_sentences(
        args.dataset_file,
        train_val_splits,
        # args.max_sentence_len,
        save_path=sample_list_dir)

    # Create the dataset and data loader instance
    train_dataset = ANetDataset(args.feature_root,
                                n_train_videos,
                                args.train_splits,
                                args.slide_window_size,
                                args.dur_file,
                                args.kernel_list,
                                text_proc, raw_data,
                                args.pos_thresh, args.neg_thresh,
                                args.stride_factor,
                                args.dataset,
                                args.sampling_sec,
                                save_samplelist=args.save_train_samplelist,
                                load_samplelist=args.load_train_samplelist,
                                sample_list_dir=args.train_samplelist_path,
                                )

    valid_dataset = ANetDataset(args.feature_root,
                                n_val_videos,
                                args.val_splits,
                                args.slide_window_size,
                                args.dur_file,
                                args.kernel_list,
                                text_proc, raw_data,
                                args.pos_thresh, args.neg_thresh,
                                args.stride_factor,
                                args.dataset,
                                args.sampling_sec,
                                save_samplelist=args.save_valid_samplelist,
                                load_samplelist=args.load_valid_samplelist,
                                sample_list_dir=args.valid_samplelist_path,
                                )

    # if text_proc is not None:
    #     exit()

    if not train_dataset.samples_loaded:
        train_dataset.get_samples(args.n_proc)
    if not valid_dataset.samples_loaded:
        valid_dataset.get_samples(args.n_proc)


    from datetime import timedelta

    if args.distributed and args.cuda:
        from urllib.parse import urlparse

        k = urlparse(args.dist_url)
        print()
        print(f'dist_url scheme: {k.scheme}')
        print(f'dist_url path: {k.path}')
        print()

        if k.scheme == 'file' and os.path.exists(k.path):
            print(f'removing existing dist_url path: {k.path}')
            os.remove(k.path)
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=0,
                                timeout=timedelta(seconds=10)
                                )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=anet_collate_fn)

    return train_loader, valid_loader, text_proc, train_sampler


def get_model(text_proc, args):
    """

    :param text_proc:
    :param TrainParams args:
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
                               nsamples=args.train_sample,
                               kernel_list=args.kernel_list,
                               stride_factor=args.stride_factor,
                               learn_mask=args.mask_weight > 0)

    # Initialize the networks and the criterion
    if len(args.start_from) > 0:
        print("Initializing weights from {}".format(args.start_from))
        model.load_state_dict(torch.load(args.start_from,
                                         map_location=lambda storage, location: storage))

    # Ship the model to GPU, maybe
    if args.cuda:
        if args.distributed:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
        else:
            model = torch.nn.DataParallel(model).cuda()
        # elif torch.cuda.device_count() > 1:
        #     model = torch.nn.DataParallel(model).cuda()
        # else:
        #     model.cuda()
    return model


def main():
    args = get_args()  # type: TrainParams

    print(f'args.resume: {args.resume}')

    # dist parallel, optional
    args.distributed = args.world_size > 1
    # args.distributed = 1

    # args = TrainParams()
    # paramparse.process(args)

    # arguments inspection

    """
    slide_window_size is in units of SAMPLED frames rather than original ones
    this is also a misnomer since there is an implicit and 
    mind bogglingly annoying assumption underlying this entire gunky operation that none of the 
    training or testing videos exceed this length otherwise the excess part will be ignored rather than 
    any kind of actual sliding window operation happening to process the long video piecewise
    """
    assert (args.slide_window_size >= args.slide_window_stride)
    # assert (args.sampling_sec == 0.5)  # attention! sampling_sec is hard coded as 0.5

    if not args.train_samplelist_path:
        args.train_samplelist_path = linux_path(args.ckpt, f"{args.train_splits[0],}_samples")

    # if not args.train_sentence_dict_path:
    #     args.train_sentence_dict_path = linux_path(args.ckpt, "train_sentence_dict.pkl")

    if not args.valid_samplelist_path:
        args.valid_samplelist_path = linux_path(args.ckpt, f"{args.val_splits[0],}_samples")

    # if not args.valid_sentence_dict_path:
    #     args.valid_sentence_dict_path = linux_path(args.ckpt, "valid_sentence_dict.pkl")

    print(f'save_valid_samplelist: {args.save_valid_samplelist}')
    print(f'save_train_samplelist: {args.save_train_samplelist}')
    print(f'valid_samplelist_path: {args.valid_samplelist_path}')
    print(f'train_samplelist_path: {args.train_samplelist_path}')

    if args.db_root:
        args.feature_root = linux_path(args.db_root, args.feature_root)
        args.dataset_file = linux_path(args.db_root, args.dataset_file)
        args.dur_file = linux_path(args.db_root, args.dur_file)

    print('loading dataset')
    train_loader, valid_loader, text_proc, train_sampler = get_dataset(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.ckpt, exist_ok=True)

    print('building model')
    model = get_model(text_proc, args)

    # filter params that don't require gradient (credit: PyTorch Forum issue 679)
    # smaller learning rate for the decoder
    if args.optim == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate, betas=(args.alpha, args.beta), eps=args.epsilon)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            args.learning_rate,
            weight_decay=1e-5,
            momentum=args.alpha,
            nesterov=True
        )
    else:
        raise NotImplementedError

    # learning rate decay every 1 epoch
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.reduce_factor,
                                               patience=args.patience_epoch,
                                               verbose=True)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, 0.6)

    # Number of parameter blocks in the network
    print("# of param blocks: {}".format(str(len(list(model.parameters())))))

    best_train_loss = float('inf')
    best_train_loss_epoch = None

    best_valid_loss = float('inf')
    best_valid_loss_epoch = None

    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom()
        vis_window = {'iter': None,
                      'loss': None}
    else:
        vis, vis_window = None, None

    all_eval_losses = []
    all_cls_losses = []
    all_reg_losses = []
    all_sent_losses = []
    all_mask_losses = []
    all_training_losses = []

    tb_path = linux_path(args.ckpt, 'tb')
    os.makedirs(tb_path, exist_ok=1)

    print(f'saving tensorboard log to: {tb_path}')

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=tb_path)

    start_epoch = 0

    if args.resume:
        checkpoint, ckpt_epoch = get_latest_checkpoint(args.ckpt, ignore_missing=True)
        if checkpoint is not None:
            print(f"loading weights from {checkpoint}")
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict)
            start_epoch = ckpt_epoch + 1

    for train_epoch in range(start_epoch, args.max_epochs):
        t_epoch_start = time.time()
        # print('Epoch: {}'.format(train_epoch))

        if args.distributed:
            train_sampler.set_epoch(train_epoch)

        train_loss_dict = train(train_epoch, model, optimizer, train_loader,
                                vis, vis_window, args)

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

        if train_epoch % args.save_checkpoint_every == 0 or train_epoch == args.max_epochs:
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                del_epoch = train_epoch - args.keep_checkpoints
                del_ckpt = os.path.join(args.ckpt, f'epoch_{del_epoch}.pth')
                if del_epoch >= 0 and os.path.exists(del_ckpt):
                    os.remove(del_ckpt)

                ckpt = os.path.join(args.ckpt, f'epoch_{train_epoch}.pth')
                print(f'saving regular checkpoint: {ckpt}')

                torch.save(model.state_dict(), ckpt)

        (valid_loss, val_cls_loss,
         val_reg_loss, val_sent_loss, val_mask_loss) = valid(train_epoch, model, valid_loader, args)

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

        if args.enable_visdom:
            if vis_window['loss'] is None:
                if not args.distributed or (args.distributed and dist.get_rank() == 0):
                    vis_window['loss'] = vis.line(
                        X=np.tile(np.arange(len(all_eval_losses)),
                                  (6, 1)).T,
                        Y=np.column_stack((np.asarray(all_training_losses),
                                           np.asarray(all_eval_losses),
                                           np.asarray(all_cls_losses),
                                           np.asarray(all_reg_losses),
                                           np.asarray(all_sent_losses),
                                           np.asarray(all_mask_losses))),
                        opts=dict(title='Loss',
                                  xlabel='Validation Iter',
                                  ylabel='Loss',
                                  legend=['train',
                                          'dev',
                                          'dev_cls',
                                          'dev_reg',
                                          'dev_sentence',
                                          'dev_mask']))
            else:
                if not args.distributed or (
                        args.distributed and dist.get_rank() == 0):
                    vis.line(
                        X=np.tile(np.arange(len(all_eval_losses)),
                                  (6, 1)).T,
                        Y=np.column_stack((np.asarray(all_training_losses),
                                           np.asarray(all_eval_losses),
                                           np.asarray(all_cls_losses),
                                           np.asarray(all_reg_losses),
                                           np.asarray(all_sent_losses),
                                           np.asarray(all_mask_losses))),
                        win=vis_window['loss'],
                        opts=dict(title='Loss',
                                  xlabel='Validation Iter',
                                  ylabel='Loss',
                                  legend=['train',
                                          'dev',
                                          'dev_cls',
                                          'dev_reg',
                                          'dev_sentence',
                                          'dev_mask']))

        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            best_train_loss_epoch = train_epoch
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                checkpoint, epoch = get_latest_checkpoint(args.ckpt, 'best_train_model_', True)
                if checkpoint is not None:
                    os.remove(checkpoint)
                ckpt_path = os.path.join(args.ckpt, f'best_train_model_{train_epoch}.pth')
                torch.save(model.state_dict(), ckpt_path)
            print('*' * 5)
            print(f'Better training loss {epoch_loss:.4f} found, save model')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_loss_epoch = train_epoch
            if (args.distributed and dist.get_rank() == 0) or not args.distributed:
                checkpoint, epoch = get_latest_checkpoint(args.ckpt, 'best_val_model_', True)
                if checkpoint is not None:
                    os.remove(checkpoint)
                ckpt_path = os.path.join(args.ckpt, f'best_val_model_{train_epoch}.pth')
                torch.save(model.state_dict(), ckpt_path)
            print('*' * 5)
            print('Better validation loss {:.4f} found, save model'.format(valid_loss))

        # save eval and train losses
        if (args.distributed and dist.get_rank() == 0) or not args.distributed:
            torch.save({'train_loss': all_training_losses,
                        'eval_loss': all_eval_losses,
                        'eval_cls_loss': all_cls_losses,
                        'eval_reg_loss': all_reg_losses,
                        'eval_sent_loss': all_sent_losses,
                        'eval_mask_loss': all_mask_losses,
                        }, os.path.join(args.ckpt, 'model_losses.pth'))

        # learning rate decay
        scheduler.step(valid_loss)

        # all other process wait for the 1st process to finish
        # if args.distributed:
        #     dist.barrier()

        print('-' * 80)
        print('Epoch {} summary'.format(train_epoch))
        print('Train loss: {:.4f}, val loss: {:.4f}, Time: {:.4f}s'.format(
            epoch_loss, valid_loss, time.time() - t_epoch_start
        ))
        print(f'best_train_loss: {best_train_loss:.4f} '
              f'in epoch: {best_train_loss_epoch:.4f}')

        print(f'val_cls: {val_cls_loss:.4f}, '
              f'val_reg: {val_reg_loss:.4f}, '
              f'val_sentence: {val_sent_loss:.4f}, '
              f'val mask: {val_mask_loss:.4f}')

        print(f'best_valid_loss: {best_valid_loss:.4f} '
              f'in epoch: {best_valid_loss_epoch:.4f}')

        print('-' * 80)


def train(epoch, model, optimizer, train_loader, vis, vis_window,
          # writer,
          args):
    model.train()  # training mode
    train_loss = []
    train_cls_loss = []
    train_reg_loss = []
    train_sent_loss = []
    train_scst_loss = []
    train_mask_loss = []

    nbatches = len(train_loader)
    # t_iter_start = time.time()

    pbar = tqdm(train_loader, total=nbatches, ncols=120)

    sample_prob = min(args.sample_prob, int(epoch / 5) * 0.05)
    for train_iter, data in enumerate(pbar):
        global_iter = epoch * nbatches + train_iter

        (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch, times) = data
        load_t, torch_t, collate_t = times
        pbar.set_description(f'training epoch {epoch} times: {load_t:.2f}, {torch_t:.2f}, {collate_t:.2f}')

        # img_batch = Variable(img_batch)
        # tempo_seg_pos = Variable(tempo_seg_pos)
        # tempo_seg_neg = Variable(tempo_seg_neg)
        # sentence_batch = Variable(sentence_batch)

        if args.cuda:
            img_batch = img_batch.cuda()
            tempo_seg_neg = tempo_seg_neg.cuda()
            tempo_seg_pos = tempo_seg_pos.cuda()
            sentence_batch = sentence_batch.cuda()

        t_model_start = time.time()
        (pred_score, gt_score,
         pred_offsets, gt_offsets,
         pred_sentence, gt_sent,
         scst_loss, mask_loss) = model(img_batch, tempo_seg_pos,
                                       tempo_seg_neg, sentence_batch,
                                       sample_prob, args.stride_factor,
                                       scst=args.scst_weight > 0,
                                       gated_mask=args.gated_mask)

        cls_loss = model.module.bce_loss(pred_score, gt_score) * args.cls_weight
        reg_loss = model.module.reg_loss(pred_offsets, gt_offsets) * args.reg_weight
        sent_loss = F.cross_entropy(pred_sentence, gt_sent) * args.sent_weight

        total_loss = cls_loss + reg_loss + sent_loss

        if scst_loss is not None:
            scst_loss *= args.scst_weight
            total_loss += scst_loss

            train_scst_loss.append(scst_loss.data.item())
            # writer.add_scalar('train_iter/scst_loss', train_scst_loss[-1], global_iter)

        if mask_loss is not None:
            mask_loss = args.mask_weight * mask_loss
            total_loss += mask_loss

            train_mask_loss.append(mask_loss.data.item())
            # writer.add_scalar('train_iter/mask_loss', train_mask_loss[-1], global_iter)

        else:
            mask_loss = cls_loss.new(1).fill_(0)

        optimizer.zero_grad()
        total_loss.backward()

        # enable the clipping for zero mask loss training
        total_grad_norm = clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                          args.grad_norm)
        # writer.add_scalar('train_iter/grad_norm', float(total_grad_norm), global_iter)

        optimizer.step()

        train_cls_loss.append(cls_loss.data.item())
        # writer.add_scalar('train_iter/cls_loss', train_cls_loss[-1], global_iter)

        train_reg_loss.append(reg_loss.data.item())
        # writer.add_scalar('train_iter/reg_loss', train_reg_loss[-1], global_iter)

        train_sent_loss.append(sent_loss.data.item())
        # writer.add_scalar('train_iter/sent_loss', train_sent_loss[-1], global_iter)

        train_loss.append(total_loss.data.item())
        # writer.add_scalar('train_iter/oss', train_loss[-1], global_iter)

        if args.enable_visdom:
            if vis_window['iter'] is None:
                if not args.distributed or (
                        args.distributed and dist.get_rank() == 0):
                    vis_window['iter'] = vis.line(
                        X=np.arange(epoch * nbatches + train_iter, epoch * nbatches + train_iter + 1),
                        Y=np.asarray(train_loss),
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss')
                    )
            else:
                if not args.distributed or (
                        args.distributed and dist.get_rank() == 0):
                    vis.line(
                        X=np.arange(epoch * nbatches + train_iter, epoch * nbatches + train_iter + 1),
                        Y=np.asarray([np.mean(train_loss)]),
                        win=vis_window['iter'],
                        opts=dict(title='Training Loss',
                                  xlabel='Training Iteration',
                                  ylabel='Loss'),
                        update='append'
                    )

        # t_model_end = time.time()
        # print('iter: [{}/{}], training loss: {:.4f}, '
        #       'class: {:.4f}, '
        #       'reg: {:.4f}, sentence: {:.4f}, '
        #       'mask: {:.4f}, '
        #       'grad norm: {:.4f} '
        #       'data time: {:.4f}s, total time: {:.4f}s'.format(
        #     train_iter, nbatches, total_loss.data.item(), cls_loss.data.item(),
        #     reg_loss.data.item(), sent_loss.data.item(), mask_loss.data.item(),
        #     total_grad_norm,
        #     t_model_start - t_iter_start,
        #     t_model_end - t_iter_start
        # ), end='\r')
        # t_iter_start = time.time()

    loss_dict = {
        'loss': train_loss,
        'cls_loss': train_cls_loss,
        'reg_loss': train_reg_loss,
        'sent_loss': train_sent_loss,
        'mask_loss': train_mask_loss,
        'scst_loss': train_scst_loss,
    }
    return loss_dict


def valid(epoch, model, loader,
          # writer,
          args):
    model.eval()
    valid_loss = []
    val_cls_loss = []
    val_reg_loss = []
    val_sent_loss = []
    val_mask_loss = []

    nbatches = len(loader)
    pbar = tqdm(loader, total=nbatches, ncols=120)

    for val_iter, data in enumerate(pbar):
        global_iter = epoch * nbatches + val_iter

        (img_batch, tempo_seg_pos, tempo_seg_neg, sentence_batch, times) = data

        load_t, torch_t, collate_t = times
        pbar.set_description(f'validation epoch {epoch} times: {load_t:.3f},{torch_t:.3f},{collate_t:.3f}')

        with torch.no_grad():
            # img_batch = Variable(img_batch)
            # tempo_seg_pos = Variable(tempo_seg_pos)
            # tempo_seg_neg = Variable(tempo_seg_neg)
            # sentence_batch = Variable(sentence_batch)

            if args.cuda:
                img_batch = img_batch.cuda()
                tempo_seg_neg = tempo_seg_neg.cuda()
                tempo_seg_pos = tempo_seg_pos.cuda()
                sentence_batch = sentence_batch.cuda()

            (pred_score, gt_score,
             pred_offsets, gt_offsets,
             pred_sentence, gt_sent,
             _, mask_loss) = model(img_batch, tempo_seg_pos,
                                   tempo_seg_neg, sentence_batch,
                                   stride_factor=args.stride_factor,
                                   gated_mask=args.gated_mask)

            cls_loss = model.module.bce_loss(pred_score, gt_score) * args.cls_weight
            reg_loss = model.module.reg_loss(pred_offsets, gt_offsets) * args.reg_weight
            sent_loss = F.cross_entropy(pred_sentence, gt_sent) * args.sent_weight

            total_loss = cls_loss + reg_loss + sent_loss

            if mask_loss is not None:
                mask_loss = args.mask_weight * mask_loss
                total_loss += mask_loss
            else:
                mask_loss = cls_loss.new(1).fill_(0)

            val_cls_loss.append(cls_loss.data.item())
            # writer.add_scalar('val_iter/cls_loss', val_cls_loss[-1], global_iter)

            val_reg_loss.append(reg_loss.data.item())
            # writer.add_scalar('val_iter/reg_loss', val_reg_loss[-1], global_iter)

            val_sent_loss.append(sent_loss.data.item())
            # writer.add_scalar('val_iter/sent_loss', val_sent_loss[-1], global_iter)

            val_mask_loss.append(mask_loss.data.item())
            # writer.add_scalar('val_iter/mask_loss', val_sent_loss[-1], global_iter)

            valid_loss.append(total_loss.data.item())
            # writer.add_scalar('val_iter/loss', valid_loss[-1], global_iter)

    return (np.mean(valid_loss), np.mean(val_cls_loss),
            np.mean(val_reg_loss), np.mean(val_sent_loss), np.mean(val_mask_loss))


if __name__ == "__main__":
    main()
