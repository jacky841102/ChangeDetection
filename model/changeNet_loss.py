import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
import pdb


def down_sample(depth_image):
    return depth_image[:, :, ::2, ::2]

def build_loss_with_mask(predict_change, groundtruth_change, groundtruth_depth):

    # clamp depth_image between 0~50.0m and get inverse depth
    vaild_depth_mask = groundtruth_depth > 0.1

    # get multi resolution depth maps
    change_image1 = groundtruth_change
    change_image2 = down_sample(change_image1)
    change_image3 = down_sample(change_image2)
    change_image4 = down_sample(change_image3)

    depth_mask1 = vaild_depth_mask
    depth_mask2 = down_sample(depth_mask1)
    depth_mask3 = down_sample(depth_mask2)
    depth_mask4 = down_sample(depth_mask3)

    # build depth image loss
    loss_change1 = F.binary_cross_entropy(predict_change[0], change_image1, reduction='none')
    loss_change2 = F.binary_cross_entropy(predict_change[1], change_image2, reduction='none')
    loss_change3 = F.binary_cross_entropy(predict_change[2], change_image3, reduction='none')
    loss_change4 = F.binary_cross_entropy(predict_change[3], change_image4, reduction='none')
    # pdb.set_trace()
    return loss_change1[depth_mask1].mean() + loss_change2[depth_mask2].mean(
    ) + loss_change3[depth_mask3].mean() + loss_change4[depth_mask4].mean()