import sys

sys.path.append('/home/abhineet/isl_labeling_tool/deep_mdp')

from input import Input
from objects import Annotations
from data import Data


class ExtractFeatureParams:
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
        assuming fps=30 gives interval=15 frames corresponding to 0.5 seconds
        this is the number of sampled frames, i.e. number of frames represented by a single feature in the network input
        """
        self.interval = 15
        self.new_size = (340, 256)

        self.input = Input.Params(source_type=-1, batch_mode=False)
        self.data = Data.Params()
        self.ann = Annotations.Params()

        self.n_proc = 1
        self.n_gpu = -1
        self.win_id = 'x99'
        self.pane_id = 6
