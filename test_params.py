from paramparse import CFG


class TestParams(CFG):
    """
    :ivar attn_dropout:
    :type attn_dropout: float

    :ivar batch_size: what is the batch size in number of images per batch? (there will be x seq_per_img sentences)
    :type batch_size: int

    :ivar cap_dropout:
    :type cap_dropout: float

    :ivar cfgs_file: dataset specific settings. anet | yc2
    :type cfgs_file: str

    :ivar cuda: use gpu
    :type cuda: bool

    :ivar d_hidden:
    :type d_hidden: int

    :ivar d_model: size of the rnn in number of hidden nodes in each layer
    :type d_model: int

    :ivar dataset: which dataset to use. two options: anet | yc2
    :type dataset: str

    :ivar dataset_file:
    :type dataset_file: str

    :ivar densecap_eval_file:
    :type densecap_eval_file: str

    :ivar densecap_references:
    :type densecap_references: str

    :ivar dur_file:
    :type dur_file: str

    :ivar feature_root: the feature root
    :type feature_root: str

    :ivar gated_mask:
    :type gated_mask: bool

    :ivar id: an id identifying this run/job. used in cross-val and appended when writing progress files
    :type id: str

    :ivar image_feat_size: the encoding size of the image feature
    :type image_feat_size: int

    :ivar in_emb_dropout:
    :type in_emb_dropout: float

    :ivar kernel_list:
    :type kernel_list: list

    :ivar learn_mask:
    :type learn_mask: bool

    :ivar max_prop_num: the maximum number of proposals per video
    :type max_prop_num: int

    :ivar max_sentence_len:
    :type max_sentence_len: int

    :ivar min_prop_before_nms: the minimum number of proposals per video
    :type min_prop_before_nms: int

    :ivar min_prop_num: the minimum number of proposals per video
    :type min_prop_num: int

    :ivar n_heads:
    :type n_heads: int

    :ivar n_layers: number of layers in the sequence model
    :type n_layers: int

    :ivar num_workers:
    :type num_workers: int

    :ivar pos_thresh:
    :type pos_thresh: float

    :ivar sampling_sec: sample frame (RGB and optical flow) with which time interval
    :type sampling_sec: float

    :ivar slide_window_size: the (temporal) size of the sliding window
    :type slide_window_size: int

    :ivar ckpt: path to a model checkpoint to initialize model weights from. Empty = dont
    :type ckpt: str

    :ivar stride_factor: the proposal temporal conv kernel stride is determined by math.ceil(kernel_len/stride_factor)
    :type stride_factor: int

    :ivar val_data_folder: validation data folder
    :type val_data_folder: str

    :ivar vis_emb_dropout:
    :type vis_emb_dropout: float
    """

    def __init__(self):
        CFG.__init__(self)

        self.cfgs_file = ''
        self.db_root = ''
        self.feature_root = ''

        self.kernel_list = []

        self.gpu = ''
        self.attn_dropout = 0.2
        self.batch_size = 1
        self.cap_dropout = 0.2
        self.cuda = True
        self.fps = 30
        self.d_rgb = 2048
        self.d_flow = 1024
        self.d_hidden = 2048
        self.d_model = 1024
        self.dataset = ''
        self.dataset_file = ''
        self.densecap_eval_file = ''
        self.densecap_references = ''
        self.dur_file = ''
        self.gated_mask = False
        self.id = ''
        self.image_feat_size = 3072
        self.in_emb_dropout = 0.1
        self.learn_mask = False
        # self.max_sentence_len = 20
        self.min_prop_before_nms = 200
        self.min_prop_num = 50
        self.max_prop_num = 500

        self.n_heads = 8
        self.n_layers = 2
        self.num_workers = 2
        self.pos_thresh = 0.7
        self.sampled_frames = 1
        self.sampling_sec = 0

        self.slide_window_size = 480
        # self.slide_window_stride = 20
        self.stride_factor = 100

        self.rgb_ch = 4
        self.enable_flow = 0
        self.feat_shape = []

        self.max_batches = 0

        self.ckpt = ''
        self.ckpt_name = ''
        self.sample_list_path = ''
        self.test_split = 'validation'
        self.vis_emb_dropout = 0.1
