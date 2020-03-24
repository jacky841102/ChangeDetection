import torch
import numpy as np
import pickle
import cv2
import pdb
import os
from imgaug import augmenters as iaa
from PIL import Image
from torch.utils.data import Dataset
from numpy.linalg import inv

class SyntheticDataset(Dataset):
    
    def __init__(self, split, use_augment=True):
        """
        args:
            split: "train" or "validate"
        """
        super(SyntheticDataset, self).__init__()
        self.split = split

        if self.split == "train":
            with open('pkl/synthetic_train.pkl', 'rb') as f:
                self.data = pickle.load(f)
        elif self.split == "validate":
            with open('pkl/synthetic_validate.pkl', 'rb') as f:
                self.data = pickle.load(f)
        else:
            raise ValueError("split could only be train or test")

        self.all_pairs = self.data['data_files']

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
        return len(self.all_pairs)


    def __getitem__(self, idx):
        pair = self.all_pairs[idx]
        # left_image = np.asarray(
        #         Image.open(pair['left_image']),
        #         dtype=np.float32)
        # right_image = np.asarray(
        #         Image.open(pair['right_image']),
        #         dtype=np.float32)
        depth_image = np.asarray(
                Image.open(pair['left_depth']),
                dtype=np.float32)
        mask_image = np.asarray(
                Image.open(pair['left_mask']),
                dtype=np.float32)

        left_image = cv2.imread(pair['left_image']).astype(np.float32)
        right_image = cv2.imread(pair['right_image']).astype(np.float32)

        camera_K = self.data['intrinsic'].copy()

        #resize the image into 320x256
        scale_x = 320. / left_image.shape[1]
        scale_y = 256. / left_image.shape[0]
        left_image = cv2.resize(left_image, (320, 256))
        right_image = cv2.resize(right_image, (320, 256))
        depth_image = cv2.resize(depth_image, (320, 256))
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
        
        left_in_right_Trans = pair['left2right'].copy()

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
        depth_image = np.expand_dims(depth_image, 0) / self.data['depth_scale']
        mask_image = np.expand_dims(mask_image, 0) / 255.

        left_image = left_image.astype(np.float32)
        right_image = right_image.astype(np.float32)
        depth_image = depth_image.astype(np.float32)
        mask_image = mask_image.astype(np.float32)
        KRKiUV = KRKiUV.astype(np.float32)
        KT = KT.astype(np.float32)

        #return the sample
        return {
            'left_image': left_image,
            'right_image': right_image,
            'depth_image': depth_image,
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


    loader = DataLoader(TUMDataset("validate"), batch_size=1, shuffle=True)
    for i_batch, sample_batched in enumerate(loader):

        left_image = np.array(sample_batched['left_image'])
        right_image = np.array(sample_batched['right_image'])

        left_image_cuda = sample_batched['left_image'].cuda()
        right_image_cuda = sample_batched['right_image'].cuda()
        KRKiUV_cuda_T = sample_batched['KRKiUV'].cuda()
        KT_cuda_T = sample_batched['KT'].cuda()
        depth_image_cuda = sample_batched['depth_image'].cuda()

        left_image_cuda = Variable(left_image_cuda, volatile=True)
        right_image_cuda = Variable(right_image_cuda, volatile=True)
        depth_image_cuda = Variable(depth_image_cuda, volatile=True)

        idepth_base = 1.0 / 50.0
        idepth_step = (1.0 / 0.5 - 1.0 / 50.0) / 63.0
        costvolume = Variable(
            torch.FloatTensor(left_image.shape[0], 64, left_image.shape[2],
                              left_image.shape[3]))
        image_height = 256
        image_width = 320
        batch_number = left_image.shape[0]

        normalize_base = torch.FloatTensor(
            [image_width / 2.0, image_height / 2.0])
        normalize_base = normalize_base.unsqueeze(0).unsqueeze(-1)
        normalize_base_v = Variable(normalize_base)

        KRKiUV_v = Variable(sample_batched['KRKiUV'])
        KT_v = Variable(sample_batched['KT'])
        for depth_i in range(64):
            this_depth = 1.0 / (idepth_base + depth_i * idepth_step)
            transformed = KRKiUV_v * this_depth + KT_v
            warp_uv = transformed[:, 0:2, :] / (transformed[:, 2, :]+1e-6).unsqueeze(
                1)  #shape = batch x 2 x 81920
            warp_uv = (warp_uv - normalize_base_v) / normalize_base_v
            warp_uv = warp_uv.view(
                batch_number, 2, image_width,
                image_height)  #shape = batch x 2 x width x height

            warp_uv = warp_uv.permute(0, 3, 2,
                                      1)  #shape = batch x height x width x 2
            right_image_v = Variable(sample_batched['right_image'])
            warped = F.grid_sample(right_image_v, warp_uv)
            costvolume[:, depth_i, :, :] = torch.sum(
                torch.abs(warped - Variable(sample_batched['left_image'])),
                dim=1)
            

        costvolume = F.avg_pool2d(
            costvolume,
            5,
            stride=1,
            padding=2,
            ceil_mode=False,
            count_include_pad=True)
        np_cost = costvolume.data.numpy()
        winner_takes_all = np.argmin(np_cost[0, :, :, :], axis=0)
        print(winner_takes_all.shape)

        cv2.imshow('left_image',
                img2show(np.moveaxis(left_image[0, :, :, :], 0, -1)))
        cv2.imshow('right_image',
                img2show(np.moveaxis(right_image[0, :, :, :], 0, -1)))
        cv2.imshow('depth_image', img2show(winner_takes_all))
        
        cv2.imshow('warped', img2show(
                np.moveaxis(warped.data.numpy()[0, :, :, :], 0, -1)))
        if cv2.waitKey(0) == 27:
            break