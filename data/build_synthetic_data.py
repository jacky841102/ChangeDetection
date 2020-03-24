import numpy as np
import cv2
import PIL
import pickle
import os
import sys
from numpy import random
from PIL import Image
from imgaug import augmenters as iaa
import pdb

def getWarpuv(transform, depth_np, pixel_coordinate, camera_K, camera_K_inv):
    left_in_right_Trans = transform.copy()
    left_in_right_T = left_in_right_Trans[0:3, 3]
    left_in_right_R = left_in_right_Trans[0:3, 0:3]
    KRK_i = camera_K.dot(left_in_right_R.dot(camera_K_inv))
    KT = camera_K.dot(left_in_right_T)
    KT = np.expand_dims(KT, -1)
    KRKiUV = KRK_i.dot(pixel_coordinate)
    transformed = KRKiUV * depth_np.reshape(1,-1) + KT
    warp_uv = transformed[:2, :] / (transformed[2, :]+1e-6)
    height, width = depth_np.shape[:2]
    warp_uv = warp_uv.reshape(2, height, width)
    return warp_uv

def randomPaste(foreground, background, aug_seq):
    """
    foreground: foreground image to be pasted. PIL.Image
    background: background image to be pasted on. PIL.Image
    aug_seq: augmentation seqences of imgaug package
    """
    foreground_np = np.array(foreground)
    foreground_mask = foreground_np != 0
    fh, fw = foreground_np.shape[:2]
    img = background.copy()
    
    bw, bh = img.size[:2]
    fg, fg_mask = aug_seq(images=[foreground_np], segmentation_maps=[foreground_mask])
    fg = fg[0]
    fg_mask = fg_mask[0].astype(np.uint8) * 255
    bx = random.randint(0, bw-fw-1)
    by = random.randint(0, bh-fh-1)
    img.paste(Image.fromarray(fg), (bx, by), Image.fromarray(fg_mask))

    img_np = np.array(img)
    mask_np = np.zeros_like(img_np)
    mask_np[by:by+fh, bx:bx+fw] = fg_mask[:,:,0:1]
    return img_np, mask_np

def randomPasteBoth(img_left, img_right, depth_left, depth_right,
                    foreground, aug_seq, left_in_right_Trans, right_in_left_Trans, pixel_coordinate, camera_K, camera_K_inv):
    if random.random() < 0.5:
        # paste on left
        img_left_np, mask_left_np = randomPaste(foreground, img_left, aug_seq)
        warp_uv = getWarpuv(right_in_left_Trans, depth_right, pixel_coordinate, camera_K, camera_K_inv)
        mask_right_warp_np = cv2.remap(mask_left_np, warp_uv[0].astype(np.float32),
                                  warp_uv[1].astype(np.float32), cv2.INTER_NEAREST)
        img_right_warp_np = cv2.remap(img_left_np, warp_uv[0].astype(np.float32),
                                  warp_uv[1].astype(np.float32), cv2.INTER_LINEAR)
        img_right = img_right.copy()
        mask_right_warp = Image.fromarray(mask_right_warp_np).convert("L")
        img_right_warp = Image.fromarray(img_right_warp_np)
        img_right.paste(img_right_warp, (0,0), mask_right_warp)
        img_left = Image.fromarray(img_left_np)
        return img_left, img_right, mask_left_np
    else:
        # paste on right
        img_right_np, mask_right_np = randomPaste(foreground, img_right, aug_seq)
        warp_uv = getWarpuv(left_in_right_Trans, depth_left, pixel_coordinate, camera_K, camera_K_inv)
        mask_left_warp_np = cv2.remap(mask_right_np, warp_uv[0].astype(np.float32),
                                  warp_uv[1].astype(np.float32), cv2.INTER_NEAREST)
        img_left_warp_np = cv2.remap(img_right_np, warp_uv[0].astype(np.float32),
                                  warp_uv[1].astype(np.float32), cv2.INTER_LINEAR)
        img_left = img_left.copy()
        mask_left_warp = Image.fromarray(mask_left_warp_np).convert("L")
        img_left_warp = Image.fromarray(img_left_warp_np)
        img_left.paste(img_left_warp, (0,0), mask_left_warp)
        img_right = Image.fromarray(img_right_np)
        return img_left, img_right, mask_left_warp_np

def randomPasteOne(img_left, img_right, depth_left, depth_right,
                    foreground, aug_seq, left_in_right_Trans, right_in_left_Trans, pixel_coordinate, camera_K, camera_K_inv):
    if random.random() < 0.5:
        img_left, _, mask_left = randomPasteBoth(
            img_left, img_right, depth_left, depth_right,
            foreground, aug_seq, left_in_right_Trans, right_in_left_Trans, pixel_coordinate, camera_K, camera_K_inv)
    else:
        _, img_right, mask_left = randomPasteBoth(
            img_left, img_right, depth_left, depth_right,
            foreground, aug_seq, left_in_right_Trans, right_in_left_Trans, pixel_coordinate, camera_K, camera_K_inv)
    return img_left, img_right, mask_left
    

def generate_dataset(split):
    
    with open('pkl/tum_{}.pkl'.format(split), 'rb') as f:
        synthetic_dataset = pickle.load(f)

    camera_K = synthetic_dataset['intrinsic']
    camera_K_inv = np.linalg.inv(camera_K)
    synthetic_dir = "dataset/synthetic_objects"
    paste_objects = [Image.open(os.path.join(synthetic_dir, p)).convert('RGBA') for p in os.listdir(synthetic_dir)]

    height, width = synthetic_dataset['height'], synthetic_dataset['width']
    pixel_coordinate = np.indices([height, width]).astype(np.float32)
    pixel_coordinate = np.concatenate(
        (pixel_coordinate, np.ones([1, height, width])), axis=0)
    pixel_coordinate[[0,1]] = pixel_coordinate[[1,0]]
    pixel_coordinate = np.reshape(pixel_coordinate, [3, -1])

    aug_seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.8, 1.3)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.1), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-90, 90),
            shear=(-8, 8)
        )
    ], random_order=True)

    for idx, pair in enumerate(synthetic_dataset['data_files']):
        img1 = Image.open(pair['left_image'])
        img2 = Image.open(pair['right_image'])
        left_depth = Image.open(pair['left_depth'])
        right_depth = Image.open(pair['right_depth'])

        foreground = paste_objects[random.choice(len(paste_objects))]
        sz = random.randint(75,125)
        foreground = foreground.resize((sz, sz))

        img1_np = np.array(img1)
        img2_np = np.array(img2)
        left_depth_np = np.array(left_depth) / synthetic_dataset['depth_scale']
        right_depth_np = np.array(right_depth) / synthetic_dataset['depth_scale']
        left_in_right_Trans = pair['left2right'].copy()
        right_in_left_Trans = pair['right2left'].copy()

        img1_name = 'dataset/synthetic/{}/rgb/left_{:07}.png'.format(split, idx)
        img2_name = 'dataset/synthetic/{}/rgb/right_{:07}.png'.format(split, idx)
        left_depth_name = 'dataset/synthetic/{}/depth/left_depth_{:07}.png'.format(split, idx)
        right_depth_name = 'dataset/synthetic/{}/depth/right_depth_{:07}.png'.format(split, idx)
        mask_name = 'dataset/synthetic/{}/mask/mask_{:07}.png'.format(split, idx)

        pair['left_image'] = img1_name
        pair['right_image'] = img2_name
        pair['left_depth'] = left_depth_name
        pair['right_depth'] = right_depth_name
        pair['left_mask'] = mask_name

        if idx % 10 == 0:
            print('Build synthetic data split: {}, {}/{}'.format(split, idx, len(synthetic_dataset['data_files'])))


        for _ in range(random.randint(0, 3)):
            foreground = paste_objects[random.choice(len(paste_objects))]
            sz = random.randint(75,125)
            foreground = foreground.resize((sz, sz))
            img1, img2, _ = randomPasteBoth(img1, img2, left_depth_np, right_depth_np, foreground,
                            aug_seq, left_in_right_Trans, right_in_left_Trans, pixel_coordinate,
                            camera_K, camera_K_inv)

        left_mask_np = np.zeros((height, width), dtype=bool)
        for _ in range(random.randint(1, 4)):
            foreground = paste_objects[random.choice(len(paste_objects))]
            sz = random.randint(75,125)
            foreground = foreground.resize((sz, sz))
            img1, img2, mask_np = randomPasteOne(img1, img2, left_depth_np, right_depth_np, foreground,
                            aug_seq, left_in_right_Trans, right_in_left_Trans, pixel_coordinate,
                            camera_K, camera_K_inv)
            left_mask_np = np.bitwise_or(left_mask_np, mask_np[:,:,0])

        left_mask = Image.fromarray(left_mask_np, 'L')
        img1.save(img1_name)
        img2.save(img2_name)
        left_depth.save(left_depth_name)
        right_depth.save(right_depth_name)
        left_mask.save(mask_name)
    
    with open("pkl/synthetic_{}.pkl".format(split), 'wb') as f:
        pickle.dump(synthetic_dataset, f)

if __name__ == "__main__":
    generate_dataset(sys.argv[1])