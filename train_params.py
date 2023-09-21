import os


class TrainParams:
    """
    :ivar alpha: alpha for adagrad/rmsprop/momentum/adam
    :ivar attn_dropout: None
    :ivar batch_size: what is the batch size in number of images per batch? (there will be x seq_per_img sentences)
    :ivar beta: beta used for adam
    :ivar cap_dropout: None
    :ivar cfgs_file: dataset specific settings. anet | yc2
    :ivar ckpt: folder to save checkpoints into (empty = this folder)
    :ivar cls_weight: None
    :ivar cuda: use gpu
    :ivar d_hidden: None
    :ivar d_model: size of the rnn in number of hidden nodes in each layer
    :ivar dataset: which dataset to use. two options: anet | yc2
    :ivar dataset_file: None
    :ivar dist_backend: distributed backend
    :ivar dist_url: url used to set up distributed training
    :ivar dur_file: None
    :ivar enable_visdom: None
    :ivar epsilon: epsilon that goes into denominator for smoothing
    :ivar feature_root: the feature root
    :ivar gated_mask: None
    :ivar grad_norm: Gradient clipping norm
    :ivar image_feat_size: the encoding size of the image feature
    :ivar in_emb_dropout: None
    :ivar kernel_list: None
    :ivar learning_rate: learning rate
    :ivar load_train_samplelist: None
    :ivar load_valid_samplelist: None
    :ivar loss_alpha_r: The weight for regression loss
    :ivar losses_log_every: How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)
    :ivar mask_weight: None
    :ivar max_epochs: max number of epochs to run for
    :ivar max_sentence_len: None
    :ivar n_heads: None
    :ivar n_layers: number of layers in the sequence model
    :ivar neg_thresh: None
    :ivar num_workers: None
    :ivar optim: what update to use? rmsprop|sgd|sgdmom|adagrad|adam
    :ivar patience_epoch: Epoch to wait to determine a pateau
    :ivar pos_thresh: None
    :ivar reduce_factor: Factor of learning rate reduction
    :ivar reg_weight: None
    :ivar sample_prob: probability for use model samples during training
    :ivar save_checkpoint_every: how many epochs to save a model checkpoint?
    :ivar save_train_samplelist: None
    :ivar save_valid_samplelist: None
    :ivar scst_weight: None
    :ivar seed: random number generator seed to use
    :ivar sent_weight: None
    :ivar slide_window_size: the (temporal) size of the sliding window
    :ivar start_from: path to a model checkpoint to initialize model weights from. Empty = dont
    :ivar stride_factor: the proposal temporal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)
    :ivar train_splits: training data folder
    :ivar train_sample: total number of positive+negative training samples (2*U)
    :ivar train_samplelist_path: None
    :ivar val_splits: validation data folder
    :ivar valid_batch_size: None
    :ivar valid_samplelist_path: None
    :ivar vis_emb_dropout: None
    :ivar world_size: number of distributed processes
    """

    def __init__(self):
        self.cfgs_file = ''
        self.gpu = ''
        self.alpha = 0.95
        self.attn_dropout = 0.2
        self.batch_size = 32
        self.beta = 0.999
        self.cap_dropout = 0.2
        self.ckpt = ''
        self.cls_weight = 1.0
        self.cuda = 1
        self.d_hidden = 2048
        # self.d_rgb = 2048
        # self.d_flow = 1024
        self.rgb_ch = 4

        self.fuse_conv_bn = 0
        self.feat_cfg = ''
        self.feat_ckpt = ''
        self.feat_shape = []
        self.feat_batch_size = 6
        self.feat_reduction = []
        self.mean = []
        self.std = []

        self.class_info_path = 'data/mnist_mot.txt'

        self.d_model = 1024

        self.gpu = ''

        self.dataset = ''
        self.dataset_file = ''
        self.local_rank = 0
        self.distributed = 1
        self.dist_backend = 'nccl'
        self.dist_url = 'file:///home/abhineet/dnc_nonexistent_file'
        self.dur_file = ''
        self.enable_visdom = False
        self.epsilon = 1e-08
        self.db_root = ''
        self.feature_root = ''
        self.gated_mask = 0
        self.grad_norm = 1
        self.image_feat_size = 3072
        self.in_emb_dropout = 0.1
        self.kernel_list = [1, 2, 3, 4, 5, 7, 9, 11, 15, 21,
                            29, 41, 57, 71, 111, 161, 211, 251]
        self.learning_rate = 0.1
        self.load_train_samplelist = 1
        self.load_valid_samplelist = 1
        self.loss_alpha_r = 2
        self.losses_log_every = 1
        self.mask_weight = 0.0
        self.max_epochs = 20
        # self.max_sentence_len = 200
        self.n_heads = 8
        self.n_layers = 2
        self.neg_thresh = 0.3
        self.num_workers = 1
        self.optim = 'sgd'
        self.patience_epoch = 1
        self.pos_thresh = 0.7
        self.reduce_factor = 0.5
        self.reg_weight = 10
        self.sample_prob = 0
        self.vis_fps = 10.0
        self.fps = 30.0
        self.sampled_frames = 1.0
        self.keep_checkpoints = 3
        self.save_checkpoint_every = 1
        self.validate_every = 1
        self.save_train_samplelist = 1
        self.save_valid_samplelist = 1
        self.resume = 0
        self.scst_weight = 0.0
        self.seed = 213
        self.sent_weight = 0.25

        self.slide_window_size = 480
        # self.slide_window_stride = 20
        self.stride_factor = 100

        self.enable_flow = 0

        self.max_vis_traj = 50
        self.vis_batch_id = -1
        self.vis_sample_id = -1
        self.vis_from = 0
        self.vis = 2
        self.vocab_fmt = 0
        self.max_diff = 0
        self.grid_res = [32, 32]

        self.img_dir_name = 'Images'

        self.min_prop_before_nms = 200
        self.min_prop_num = 50
        self.max_prop_num = 500

        self.start_from = ''

        self.train_splits = ['training', ]
        self.train_sample = 20
        self.train_samplelist_path = ''
        # self.train_sentence_dict_path = ''

        self.val_splits = ['validation', ]
        self.valid_batch_size = 0
        self.valid_samplelist_path = ''
        # self.valid_sentence_dict_path = ''

        self.vis_emb_dropout = 0.1
        self.world_size = 1
        self.n_proc = 1

        self.densecap_references = []


def get_args():
    import argparse
    import yaml

    from dnc_data.utils import update_values

    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--cfgs_file', default='cfgs/anet.yml', type=str, help='dataset specific settings. anet | yc2')
    parser.add_argument('--class_info_path', default='data/mnist_mot.txt', type=str)
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--dataset', default='', type=str, help='which dataset to use')
    parser.add_argument('--dataset_file', default='', type=str)
    parser.add_argument('--db_root', default='', type=str, help='the dataset root')
    parser.add_argument('--feature_root', default='', type=str, help='the feature root')
    parser.add_argument('--dur_file', default='', type=str)
    parser.add_argument('--train_splits', default=['training'], type=str, nargs='+', help='training data folder')
    parser.add_argument('--val_splits', default=['validation'], help='validation data folder')

    parser.add_argument('--save_train_samplelist', default=1, type=int)
    parser.add_argument('--load_train_samplelist', default=1, type=int)
    parser.add_argument('--train_samplelist_path', type=str, default='')
    # parser.add_argument('--train_sentence_dict_path', type=str, default='')
    parser.add_argument('--save_valid_samplelist', default=1, type=int)
    parser.add_argument('--load_valid_samplelist', default=1, type=int)
    parser.add_argument('--valid_samplelist_path', type=str, default='')
    # parser.add_argument('--valid_sentence_dict_path', type=str, default='')

    parser.add_argument('--start_from', default='',
                        help='path to a model checkpoint to initialize model weights from. Empty = dont')
    # parser.add_argument('--max_sentence_len', default=200, type=int)
    parser.add_argument('--num_workers', default=1, type=int)

    # Model settings: General
    # parser.add_argument('--d_rgb', default=2048, type=int)
    # parser.add_argument('--d_flow', default=1024, type=int)
    parser.add_argument('--d_model', default=1024, type=int,
                        help='size of the rnn in number of hidden nodes in each layer')
    parser.add_argument('--fuse_conv_bn', default=0, type=int)
    parser.add_argument('--feat_cfg', default='', type=str)
    parser.add_argument('--feat_ckpt', default='', type=str)
    parser.add_argument('--feat_reduction', default=[], type=str, nargs='+')
    parser.add_argument('--feat_shape', default=[], type=int, nargs='+')
    parser.add_argument('--feat_batch_size', default=2, type=int)
    parser.add_argument('--mean', default=[], type=float, nargs='+')
    parser.add_argument('--std', default=[], type=float, nargs='+')

    parser.add_argument('--rgb_ch', default=4, type=int)
    parser.add_argument('--d_hidden', default=2048, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--in_emb_dropout', default=0.1, type=float)
    parser.add_argument('--attn_dropout', default=0.2, type=float)
    parser.add_argument('--vis_emb_dropout', default=0.1, type=float)
    parser.add_argument('--cap_dropout', default=0.2, type=float)
    parser.add_argument('--image_feat_size', default=3072, type=int, help='the encoding size of the image feature')
    parser.add_argument('--n_layers', default=2, type=int, help='number of layers in the sequence model')
    parser.add_argument('--train_sample', default=20, type=int,
                        help='total number of positive+negative training samples (2*U)')
    parser.add_argument('--sample_prob', default=0, type=float,
                        help='probability for use model samples during training')

    # Model settings: Proposal and mask
    parser.add_argument('--slide_window_size', default=480, type=int, help='the (temporal) size of the sliding window')
    # parser.add_argument('--slide_window_stride', default=20, type=int, help='the step size of the sliding window')
    parser.add_argument('--vis_fps', default=10, type=float)
    parser.add_argument('--fps', default=30.0, type=float)
    parser.add_argument('--sampled_frames', default=1.0, type=float)

    parser.add_argument('--kernel_list', default=[1, 2, 3, 4, 5, 7, 9, 11, 15, 21, 29, 41, 57, 71, 111, 161, 211, 251],
                        type=int, nargs='+')

    parser.add_argument('--grid_res', default=[32, 32], type=int, nargs='+')

    parser.add_argument('--pos_thresh', default=0.7, type=float)
    parser.add_argument('--neg_thresh', default=0.3, type=float)
    parser.add_argument('--stride_factor', default=100, type=int,
                        help='the proposal temporal conv kernel stride is determined by math.ceil('
                             'kernel_len/stride_factor)')
    parser.add_argument('--enable_flow', default=0, type=int)

    parser.add_argument('--img_dir_name', default='Images', type=str)
    parser.add_argument('--ext', default='mp4', type=str)

    parser.add_argument('--max_vis_traj', default=50, type=int)
    parser.add_argument('--vis_from', default=10, type=int)
    parser.add_argument('--vis_batch_id', default=-1, type=int)
    parser.add_argument('--vis_sample_id', default=-1, type=int)
    parser.add_argument('--vis', default=2, type=int)
    parser.add_argument('--vocab_fmt', default=0, type=int)
    parser.add_argument('--max_diff', default=0, type=int)

    parser.add_argument('--min_prop_before_nms', default=200, type=int)
    parser.add_argument('--min_prop_num', default=50, type=int)
    parser.add_argument('--max_prop_num', default=500, type=int)

    # Optimization: General
    parser.add_argument('--max_epochs', default=1000, type=int, help='max number of epochs to run for')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--valid_batch_size', default=0, type=int)

    parser.add_argument('--cls_weight', default=1.0, type=float)
    parser.add_argument('--reg_weight', default=10, type=float)
    parser.add_argument('--sent_weight', default=0.25, type=float)

    parser.add_argument('--scst_weight', default=0.0, type=float)
    parser.add_argument('--mask_weight', default=0.0, type=float)

    # Optimization
    parser.add_argument('--optim', default='sgd', help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate')
    parser.add_argument('--alpha', default=0.95, type=float, help='alpha for adagrad/rmsprop/momentum/adam')
    parser.add_argument('--beta', default=0.999, type=float, help='beta used for adam')
    parser.add_argument('--epsilon', default=1e-8, help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--loss_alpha_r', default=2, type=int, help='The weight for regression loss')
    parser.add_argument('--patience_epoch', default=1, type=int, help='Epoch to wait to determine a pateau')
    parser.add_argument('--reduce_factor', default=0.5, type=float, help='Factor of learning rate reduction')
    parser.add_argument('--grad_norm', default=1, type=float, help='Gradient clipping norm')
    parser.add_argument('--local_rank', type=int, default=0)

    # Data parallel
    parser.add_argument('--dist_url',
                        # default='file:///home/abhineet/nonexistent_file',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--distributed', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')

    parser.add_argument('--n_proc', default=1, type=int, help='number of processes to use while generating samples')

    # Evaluation/Checkpointing
    parser.add_argument('--validate_every', default=1, type=int)

    parser.add_argument('--save_checkpoint_every', default=1, type=int,
                        help='how many epochs to save a model checkpoint?')
    parser.add_argument('--keep_checkpoints', default=3, type=int,
                        help='how many previous checkpoints to keep')
    parser.add_argument('--ckpt', default='',
                        help='folder to save checkpoints into (empty = this folder)')
    parser.add_argument('--losses_log_every', default=1, type=int,
                        help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')
    parser.add_argument('--seed', default=213, type=int, help='random number generator seed to use')
    parser.add_argument('--resume', default=1, type=int, help='resume training')
    parser.add_argument('--enable_visdom', default=0, type=int, help='enable_visdom')
    parser.add_argument('--cuda', default=1, type=int, help='use cuda')
    parser.add_argument('--gated_mask', default=0, type=int)

    args = parser.parse_args()

    with open(args.cfgs_file, 'r') as handle:
        options_yaml = yaml.safe_load(handle)
    update_values(options_yaml, vars(args))
    # print(args)

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
