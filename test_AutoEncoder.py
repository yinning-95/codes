import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from threeD_model_final import SpatialSoftmax3D, tile2openpose_conv3d
from model_SetNet_upgrade import tile2pose_se
from model_AutoEncoder import tile2openpose_conv3d_ae
from threeD_dataLoader_batch import sample_data, ValSampleData
from threeD_dataLoader import sample_data_diffTask
import pickle
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from threeD_viz_video import generateVideo
from threeD_viz_image import generateImage

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='./train/', help='Experiment path')
parser.add_argument('--exp', type=str, default='singlePeople', help='Name of experiment')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--window', type=int, default=10, help='window around the time step')
parser.add_argument('--subsample', type=int, default=1, help='subsample tile res')
parser.add_argument('--linkLoss', type=bool, default=True, help='use link loss')
parser.add_argument('--epoch', type=int, default=100, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--ckpt', type=str, default='epoch_13_2_dis_244.47', help='loaded ckpt file')
parser.add_argument('--eval', type=bool, default=True, help='Set true if eval time')
parser.add_argument('--test_dir', type=str, default='/home/jovyan/input/dataset/Test/', help='test data path')
parser.add_argument('--exp_image', type=bool, default=False, help='Set true if export predictions as images')
parser.add_argument('--exp_video', type=bool, default=False, help='Set true if export predictions as video')
parser.add_argument('--exp_data', type=bool, default=False, help='Set true if export predictions as raw data')
parser.add_argument('--exp_L2', type=bool, default=True, help='Set true if export L2 distance')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
args = parser.parse_args()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_spatial_keypoint(keypoint):
    b = np.reshape(np.array([-100, -100, -1800]), (1, 1, 3))
    resolution = 100
    max = 19
    spatial_keypoint = keypoint * max * resolution + b
    return spatial_keypoint


def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    # mean = np.reshape(np.mean(dis, axis=0), (21,3))
    return dis


def remove_small(heatmap, threshold):
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).to(device)
    heatmap = torch.where(heatmap < threshold, z, heatmap)
    return heatmap


def check_link(min, max, keypoint):
    # print (torch.max(max), torch.min(min))
    BODY_25_pairs = np.array([
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
    [12, 13], [13, 14], [1, 0], [14, 15], [15, 16], [14, 17], [11, 18], [18, 19], [11, 20]])
    # o = torch.ones(keypoint.shape[0], keypoint.shape[1], keypoint.shape[2]).cuda()
    # keypoint = torch.where(torch.isnan(keypoint), o, keypoint)
    keypoint_output = torch.ones(keypoint.shape[0], 20).to(device)

    for f in range(keypoint.shape[0]):
        for i in range(20):
            a = keypoint[f, BODY_25_pairs[i, 0]]
            b = keypoint[f, BODY_25_pairs[i, 1]]
            s = torch.sum((a-b)**2)

            if s < min[i]:
                keypoint_output[f, i] = min[i] - s
            elif s > max[i]:
                keypoint_output[f, i] = s - max[i]
            else:
                keypoint_output[f, i] = 0
    return keypoint_output


if not os.path.exists(args.exp_dir + 'ckpts'):
    os.makedirs(args.exp_dir + 'ckpts')

if not os.path.exists(args.exp_dir + 'log'):
    os.makedirs(args.exp_dir + 'log')

if not os.path.exists(args.exp_dir + 'predictions'):
    os.makedirs(args.exp_dir + 'predictions')
    os.makedirs(args.exp_dir + 'predictions/image')
    os.makedirs(args.exp_dir + 'predictions/video')
    os.makedirs(args.exp_dir + 'predictions/L2')
    os.makedirs(args.exp_dir + 'predictions/data')

use_gpu = True
device = "cuda:1" if torch.cuda.is_available() else "cpu"  # #####

if args.linkLoss:
    link_min = pickle.load(open(args.exp_dir + '../link_min.p', "rb"))
    link_max = pickle.load(open(args.exp_dir + '../link_max.p', "rb"))

    link_min = torch.tensor(link_min, dtype=torch.float).to(device)
    link_max = torch.tensor(link_max, dtype=torch.float).to(device)

if args.eval:
    test_path = args.test_dir
    mask = []
    test_dataset = ValSampleData(test_path, args.window, mask, args.subsample)
    # test_dataset = sample_data_diffTask(test_path, args.window, args.subsample) # use this line for the diffTask test set
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print (len(test_dataset))

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    model = tile2openpose_conv3d_ae()  # model
    softmax = SpatialSoftmax3D(20, 20, 18, 21)

    model.to(device)
    softmax.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    criterion = nn.MSELoss()
    triplet_loss = nn.TripletMarginLoss(margin=4)

    epochs = -1
    if args.train_continue:
        checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.ckpt + '.path.tar')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
        epochs = checkpoint['epoch']
        loss = checkpoint['loss']
        print("ckpt loaded", loss)
        print("Now continue training")

    if args.eval:
        ckpts = ['epoch_30_1_dis_104.81']  # ##### ckpt_file
        for ckpt in ckpts:
            checkpoint = torch.load(args.exp_dir + 'ckpts1/' + ckpt + '.path.tar', map_location='cuda:1')
            para = {k.replace('module.',''):v for k,v in checkpoint['model_state_dict'].items()}
            model.load_state_dict(para)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epochs = checkpoint['epoch']
            loss = checkpoint['loss']
            print("loss:", loss)
            print("ckpt loaded:", ckpt)
            print("learning rate:", optimizer.state_dict()['param_groups'][0]['lr'])
            print("Now running on test set")
            model.eval()
            avg_val_loss = []
            avg_val_keypoint_l2_loss = []

            tactile_GT = np.empty((1, 96, 96))
            heatmap_GT = np.empty((1, 21, 20, 20, 18))
            heatmap_pred = np.empty((1, 21, 20, 20, 18))
            keypoint_GT = np.empty((1, 21, 3))
            keypoint_pred = np.empty((1, 21, 3))
            tactile_GT_v = np.empty((1, 96, 96))
            heatmap_GT_v = np.empty((1, 21, 20, 20, 18))
            heatmap_pred_v = np.empty((1, 21, 20, 20, 18))
            keypoint_GT_v = np.empty((1, 21, 3))
            keypoint_pred_v = np.empty((1, 21, 3))
            keypoint_GT_log = np.empty((1, 21, 3))
            keypoint_pred_log = np.empty((1, 21, 3))

            bar = ProgressBar(max_value=len(test_dataloader))

            c = 0
            for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
                tactile = torch.as_tensor(sample_batched[0], dtype=torch.float).to(device)
                heatmap = torch.as_tensor(sample_batched[1], dtype=torch.float).to(device)
                keypoint = torch.as_tensor(sample_batched[2], dtype=torch.float).to(device)
                tactile_frame = torch.as_tensor(sample_batched[3], dtype=torch.float).to(device)

                with torch.set_grad_enabled(False):
                    heatmap_out, tactile_out = model(tactile, device)
                    heatmap_out = heatmap_out.reshape(-1, 21, 20, 20, 18)
                    heatmap_transform = remove_small(heatmap_out.transpose(2, 3), 1e-2)
                    keypoint_out, heatmap_out2 = softmax(heatmap_transform)

                loss_link = torch.mean(check_link(link_min, link_max, keypoint_out))
                loss_heatmap = torch.mean((heatmap_transform - heatmap)**2 * (heatmap + 0.5) * 2) * 1000
                loss_tactile = criterion(tactile_out, tactile)
                heatmap_out = heatmap_transform
                loss = loss_heatmap + loss_link + loss_tactile

                # export image #
                if args.exp_image:
                    base = 0
                    imageData = [heatmap.cpu().data.numpy().reshape(-1, 21, 20, 20, 18),
                                 heatmap_out.cpu().data.numpy().reshape(-1, 1, 20, 20, 18),
                                 keypoint.cpu().data.numpy().reshape(-1, 21, 3),
                                 keypoint_out.cpu().data.numpy().reshape(-1, 21, 3),
                                 tactile_frame.cpu().data.numpy().reshape(-1, 96, 96)]

                    generateImage(imageData, args.exp_dir + 'predictions/image/', i_batch, base)

                # log data for L2 distance and video #
                if args.exp_video:
                    if 0 < i_batch < 2:  # set range
                        heatmap_GT = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
                        heatmap_pred = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
                        keypoint_GT = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                        keypoint_pred = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                        tactile_GT = np.append(tactile_GT, tactile_frame.cpu().data.numpy().reshape(-1, 96, 96), axis=0)
                if args.exp_L2:
                    keypoint_GT_log = np.append(keypoint_GT_log, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                    keypoint_pred_log = np.append(keypoint_pred_log, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)

                # save data #
                if args.exp_data:
                    heatmap_GT = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
                    heatmap_pred = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1, 21, 20, 20, 18), axis=0)
                    keypoint_GT = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                    keypoint_pred = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                    tactile_GT = np.append(tactile_GT, tactile_frame.cpu().data.numpy(), axis=0)

                    if i_batch % 20 == 0 and i_batch != 0:  # set the limit to avoid overflow
                        c += 1
                        toSave = [heatmap_GT[1:, :, :, :, :], heatmap_pred[1:, :, :, :, :],
                                  keypoint_GT[1:, :, :], keypoint_pred[1:, :, :], tactile_GT[1:, :, ]]
                        pickle.dump(toSave, open(args.exp_dir + 'predictions/data/' + ckpt + str(c) + '.p', "wb"))
                        tactile_GT = np.empty((1, 96, 96))
                        heatmap_GT = np.empty((1, 21, 20, 20, 18))
                        heatmap_pred = np.empty((1, 21, 20, 20, 18))
                        keypoint_GT = np.empty((1, 21, 3))
                        keypoint_pred = np.empty((1, 21, 3))
                avg_val_loss.append(loss.data.item())
            print("Loss:", np.mean(avg_val_loss))

            # output L2 distance #
            if args.exp_L2:
                dis = get_keypoint_spatial_dis(keypoint_GT_log[1:, :, :], keypoint_pred_log[1:, :, :])
                pickle.dump(dis, open(args.exp_dir + 'predictions/L2/' + ckpt + '.p', "wb"))
                print("keypoint_dis_saved:", dis.shape)

            # video viz #
            if args.exp_video:
                to_save = [heatmap_GT[1:, :, :, :, :], heatmap_pred[1:, :, :, :, :],
                           keypoint_GT[1:, :, :], keypoint_pred[1:, :, :], tactile_GT[1:, :, :]]
                print(to_save[0].shape, to_save[1].shape, to_save[2].shape, to_save[3].shape, to_save[4].shape)
                generateVideo(to_save, args.exp_dir + 'predictions/video/' + ckpt, heatmap=True)