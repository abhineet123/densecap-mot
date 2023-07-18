import sys

# from .config import ANET_CFG

sys.path.append('lib/dense_flow/build')

from libpydenseflow import TVL1FlowExtractor
from . import action_caffe
import numpy as np
import cv2


class FlowExtractor(object):

    def __init__(self, dev_id, bound=20):
        TVL1FlowExtractor.set_device(dev_id)
        self._et = TVL1FlowExtractor(bound)

    def extract_flow(self, frame_list, new_size=None):
        """
        This function extracts the optical flow and interleave x and y channels
        :param frame_list:
        :return:
        """
        # frame_list = [frame_list[0], frame_list[1], ]

        frame_h, frame_w = frame_list[0].shape[:2]
        # frame_str_list = [x.tobytes() for x in frame_list]
        # frame_str_list = [np.array2string(x) for x in frame_list]

        rst = self._et.extract_flow(frame_list, frame_w, frame_h)

        n_out = len(rst)
        if new_size is None:
            """
            with 6 frames in frame_list, the output from extract_flow has 5 flow images 
            since this is the number of consecutive frames pairs from which the flow can be extracted
            each of these 5 images are actually image pairs - one each for x and y directions
            these pairs are concatenated together in the following loop into a 10 x 224 x 224 matrix                        
            """
            ret = np.zeros((n_out * 2, frame_h, frame_w))
            for i in range(n_out):
                flow_img_x, flow_img_y = rst[i]

                # cv2.imshow("flow_img_x py", flow_img_x)
                # cv2.imshow("flow_img_y py", flow_img_y)
                # cv2.waitKey(0)

                # ret[2 * i, :] = np.fromstring(rst[i][0], dtype='uint8').reshape((frame_h, frame_w))
                # ret[2 * i + 1, :] = np.fromstring(rst[i][1], dtype='uint8').reshape((frame_h, frame_w))

                ret[2 * i, :] = flow_img_x
                ret[2 * i + 1, :] = flow_img_y

        else:
            ret = np.zeros((n_out * 2, new_size[1], new_size[0]))
            for i in range(n_out):
                flow_img_x, flow_img_y = rst[i]
                # cv2.imshow("flow_img_x py", flow_img_x)
                # cv2.imshow("flow_img_y py", flow_img_y)
                # cv2.waitKey(0)

                # ret[2 * i, :] = cv2.resize(np.fromstring(rst[i][0], dtype='uint8').reshape((frame_h, frame_w)),
                #                            new_size)
                # ret[2 * i + 1, :] = cv2.resize(np.fromstring(rst[i][1], dtype='uint8').reshape((frame_h, frame_w)),
                #                                new_size)

                ret[2 * i, :] = cv2.resize(flow_img_x, new_size)
                ret[2 * i + 1, :] = cv2.resize(flow_img_y, new_size)

        return ret


if __name__ == "__main__":
    import cv2

    im1 = cv2.imread('../data/img_1.jpg')
    im2 = cv2.imread('../data/img_2.jpg')

    f = FlowExtractor(0)
    flow_frames = f.extract_flow([im1, im2])
    from pylab import *

    plt.figure()
    plt.imshow(flow_frames[0])
    plt.figure()
    plt.imshow(flow_frames[1])
    plt.figure()
    plt.imshow(im1)
    plt.show()

    print(flow_frames)
