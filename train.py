import time
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import os
import pdb
from torch.autograd import Variable
from torch import Tensor
from tensorboardX import SummaryWriter
from model.changeNet_model import ChangeNet
from model.changeNet_loss import build_loss_with_mask
from data.synthetic_dataset import SyntheticDataset
from argparse import ArgumentParser
from util.visualize import np2Depth, np2Img

def main():
    epoch_num = args.epoch_num
    
    model = ChangeNet()
    model = model.cuda()
    cudnn.benchmark = True

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        SyntheticDataset("train", use_augment=True),
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        SyntheticDataset("validate", use_augment=False),
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    if args.scratch:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        writer = SummaryWriter(comment="scratch")
    else:
        pretrained_state_dict = torch.load('opensource_model.pth.tar')['state_dict']
        own_state_dict = model.state_dict()
        for name, param in pretrained_state_dict.items():
            if all([e not in name for e in ['iconv', 'upconv', 'disp']]):
                # print("Load params {}".format(name))
                own_state_dict[name].copy_(param)
                own_state_dict[name].requires_grad = False

        # optimizer
        trained_params = [param for name, param in model.named_parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(trained_params, lr=args.lr)
        writer = SummaryWriter(comment="pretrained")
    
    model_dir = 'trained_models/{}'.format(os.path.basename(writer.logdir))
    os.mkdir(model_dir)
    for epoch in range(epoch_num):
        train(model, epoch, train_loader, optimizer, writer)
        if epoch % args.eval_step == 0 or epoch == epoch_num-1:
            evaluate(model, val_loader, epoch, writer)
            torch.save(model.state_dict(), "{}/model_{}.pth".format(model_dir, epoch))

def getIoU(change, gt_change, gt_depth):
    valid_mask = gt_depth > 0.1
    change = change > 0.5
    gt_change = gt_change.astype(bool)
    intersection = np.bitwise_and(change, gt_change)
    union = np.bitwise_or(change, gt_change)
    intersection = np.sum(intersection * valid_mask, axis=(1,2))
    union = np.sum(union * valid_mask, axis=(1,2))
    iou = (intersection+1e-6)/(union+1e-6)
    return np.mean(iou)


def train(model, epoch, dataloader, optimizer, writer):
    model.train()
    for idx, batch in enumerate(dataloader):
        # get data
        left_image_cuda = batch['left_image'].cuda()
        right_image_cuda = batch['right_image'].cuda()
        KRKiUV_cuda_T = batch['KRKiUV'].cuda()
        KT_cuda_T = batch['KT'].cuda()
        depth_image_cuda = batch['depth_image'].cuda()
        mask_image_cuda = batch['mask_image'].cuda()

        left_image_cuda = Variable(left_image_cuda)
        right_image_cuda = Variable(right_image_cuda)
        depth_image_cuda = Variable(depth_image_cuda)

        optimizer.zero_grad()
        predict_changes = model(left_image_cuda, right_image_cuda,
                               KRKiUV_cuda_T, KT_cuda_T)
        loss = build_loss_with_mask(predict_changes, mask_image_cuda, depth_image_cuda)
        loss.backward()
        optimizer.step()

        global_step = len(dataloader) * epoch + idx
        if global_step % args.log_step == 0:
            change = predict_changes[0].cpu().data.numpy()[:,0,:,:]
            gt_change = np.squeeze(batch['mask_image']).data.numpy()
            gt_depth = np.squeeze(batch['depth_image']).data.numpy()
            iou = getIoU(change, gt_change, gt_depth)
            print("Epoch: {:>2}/{:>2}, batch: {:>4}/{:>4}, Global Step: {:>7}, \tLoss: {:.6f}\tIoU: {:.6f}".format(
                epoch, args.epoch_num, idx, len(dataloader), global_step, loss, iou))
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/IoU", iou, global_step)

        if global_step % args.vis_step == 0:
            # visualize the results
            for i in range(len(batch['left_image'])):
                np_left = np2Img(np.squeeze(batch['left_image'][i].numpy()), True)
                np_right = np2Img(np.squeeze(batch['right_image'][i].numpy()), True)
                
                change = predict_changes[0][i,0].cpu().data.numpy() > 0.5
                gt_change = np.squeeze(batch['mask_image'][i]).data.numpy()

                np_change = np.tile(change[:,:,np.newaxis], (1,1,3))
                np_gtchange = np.tile(gt_change[:,:,np.newaxis], (1,1,3))
                # pdb.set_trace()

                result_image = np.concatenate(
                    (np_left/255., np_right/255., np_gtchange, np_change), axis=1)
                writer.add_image("train/change_map_%d" % i, result_image, global_step, dataformats='HWC')


    
def evaluate(model, dataloader, epoch, writer):
    model.eval()
    iou_no = 0
    loss_no = 0
    denominator = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            print("Evaluate batch {}/{}".format(idx, len(dataloader)))
            left_image_cuda = batch['left_image'].cuda()
            right_image_cuda = batch['right_image'].cuda()
            KRKiUV_cuda_T = batch['KRKiUV'].cuda()
            KT_cuda_T = batch['KT'].cuda()
            depth_image_cuda = batch['depth_image'].cuda()
            mask_image_cuda = batch['mask_image'].cuda()

            left_image_cuda = Variable(left_image_cuda)
            right_image_cuda = Variable(right_image_cuda)
            depth_image_cuda = Variable(depth_image_cuda)

            predict_changes = model(left_image_cuda, right_image_cuda,
                                KRKiUV_cuda_T, KT_cuda_T)
                
            loss = build_loss_with_mask(predict_changes, mask_image_cuda, depth_image_cuda)
            change = predict_changes[0].cpu().data.numpy()[:,0,:,:]
            gt_change = np.squeeze(batch['mask_image']).data.numpy()
            gt_depth = np.squeeze(batch['depth_image']).data.numpy()
            iou = getIoU(change, gt_change, gt_depth)
            n = change.shape[0]
            iou_no += iou * n
            loss_no += loss * n
            denominator += n

        print("Validate epoch: {}\tLoss: {:.6f}\tIoU: {:.6f}".format(
            epoch, loss_no/denominator, iou_no/denominator))
        writer.add_scalar("val/IoU", iou_no/denominator, epoch)
        writer.add_scalar("val/loss", loss_no/denominator, epoch)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--epoch_num", default=20, type=int, help="number of training epochs")
    parser.add_argument("--eval_step", default=1, type=int, help="steps to evaluate")
    parser.add_argument("--log_step", default=50, type=int, help="steps to print logs")
    parser.add_argument("--vis_step", default=500, type=int, help="steps to visualize depth map")
    parser.add_argument("--scratch", action="store_true")
    args = parser.parse_args()
    main()