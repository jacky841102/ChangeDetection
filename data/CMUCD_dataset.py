import torch
import numpy as np
import pickle
import cv2
import pdb
import os
import glob
import bisect
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset
from numpy.linalg import inv

class CMUCDDataset(Dataset):
    
    def __init__(self, split, use_augment=True):
        """
        args:
            split: "train" or "validate"
        """
        super(CMUCDDataset, self).__init__()
        self.split = split

        # if self.split == "train":
        #     with open('pkl/synthetic_train.pkl', 'rb') as f:
        #         self.data = pickle.load(f)
        # elif self.split == "validate":
        #     with open('pkl/synthetic_validate.pkl', 'rb') as f:
        #         self.data = pickle.load(f)
        # else:
        #     raise ValueError("split could only be train or test")

        root = "dataset/VL_CMU_CD/raw"
        self.folders = [os.path.join(root, seq) for seq in sorted(os.listdir(root))]
        self.acc_img_num = [0]
        for folder in self.folders:
            img_num = len(glob.glob("{}/GT/*.png".format(folder)))
            self.acc_img_num.append(img_num + self.acc_img_num[-1])

        #data augment
        self.use_augment = use_augment
        if self.use_augment:
            self.img_aug = iaa.SomeOf(
                (0, 2),
                [
                    iaa.AdditiveGaussianNoise(
                        loc=0,
                        scale=(0.0,
                               0.01 * 255)),  # add gaussian noise to images
                    iaa.ContrastNormalization(
                        (0.5, 2.0),
                        per_channel=0.5),  # improve or worsen the contrast
                    iaa.Multiply((0.7, 1.3), per_channel=0.5),
                    iaa.Add((-40, 40), per_channel=0.5)
                ],
                random_order=True)
        # for image warp
        self.pixel_coordinate = np.indices([320, 256]).astype(np.float32)
        self.pixel_coordinate = np.concatenate(
            (self.pixel_coordinate, np.ones([1, 320, 256])), axis=0)
        self.pixel_coordinate = np.reshape(self.pixel_coordinate, [3, -1])


    def __len__(self):
        return self.acc_img_num[-1]


    def __getitem__(self, idx):
        folder_idx = bisect.bisect_right(self.acc_img_num, idx)-1
        seq_idx =  idx - self.acc_img_num[folder_idx]
        folder = self.folders[folder_idx]
        # pdb.set_trace()
        left_image = cv2.imread(
            "{}/RGB/1_{:02d}.png".format(folder, seq_idx)).astype(np.float32)
        right_image = cv2.imread(
            "{}/RGB/2_{:02d}.png".format(folder, seq_idx)).astype(np.float32)
        mask_image = cv2.imread(
            "{}/GT/gt{:02d}.png".format(folder, seq_idx))
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        mask_image = np.bitwise_and(mask_image > 0, mask_image < 255).astype(np.float32)

        camera_K = np.array([[525,0,512], [0,525,384], [0,0,1]], dtype=np.float32)

        #resize the image into 320x256
        scale_x = 320. / left_image.shape[1]
        scale_y = 256. / left_image.shape[0]
        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))
        # depth_image = cv2.resize(depth_image, (320, 256))
        mask_image = cv2.resize(mask_image, (320, 256), interpolation=cv2.INTER_NEAREST)
        camera_K[0, :] = camera_K[0, :] * scale_x
        camera_K[1, :] = camera_K[1, :] * scale_y
        camera_K_inv = inv(camera_K)

        if self.use_augment:
            #add the image together so that it can be augmented together,
            #otherwisw some channel maybe inconsistently augmented
            togetherImage = np.append(left_image, right_image, axis=0)
            auged_together = self.img_aug.augment_image(togetherImage)
            #sperate the image
            width = int(auged_together.shape[0] / 2)
            left_image = auged_together[:width]
            right_image = auged_together[width:]

        # left_image = (left_image - self.data['mean']
        #               ) / self.data['std']
        # right_image = (right_image - self.data['mean']
        #                ) / self.data['std']

        left_image = (left_image - 81.
                      ) / 35.
        right_image = (right_image - 81.
                       ) / 35.
        
        left_in_right_Trans = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]],dtype=np.float32)

        # demon dataset uses a differen representation
        left_in_right_T = left_in_right_Trans[0:3, 3]
        left_in_right_R = left_in_right_Trans[0:3, 0:3]
        KRK_i = camera_K.dot(left_in_right_R.dot(camera_K_inv))
        KT = camera_K.dot(left_in_right_T)
        KT = np.expand_dims(KT, -1)
        KRKiUV = KRK_i.dot(self.pixel_coordinate)

        #the image should be transformed into CxHxW
        left_image = np.moveaxis(left_image, -1, 0)
        right_image = np.moveaxis(right_image, -1, 0)
        # depth_image = np.expand_dims(depth_image, 0) / self.data['depth_scale']
        mask_image = np.expand_dims(mask_image, 0)

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        # depth_image = depth_image.astype(np.float32)
        mask_image = mask_image.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KT = KT.astype(np.float32)

        #return the sample
        return {
            'left_image': left_image,
            'right_image': right_image,
            # 'depth_image': depth_image,
            'mask_image': mask_image,
            'left2right': left_in_right_Trans,
            'KRKiUV': KRKiUV,
            'KT': KT
        }

def img2show(image):
    float_img = image.astype(float)
    print('max %f, min %f' % (float_img.max(), float_img.min()))
    float_img = (float_img - float_img.min()) / (
        float_img.max() - float_img.min()) * 255.0
    uint8_img = float_img.astype(np.uint8)
    return uint8_img

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch import Tensor
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    loader = DataLoader(CMUCDDataset("validate"), batch_size=1, shuffle=False)
    for i_batch, sample_batched in enumerate(loader):
        mask = sample_batched['mask_image'][0,0]
        mask = mask.data.numpy()

        cv2.imwrite("tmp.png", mask*255)
        break