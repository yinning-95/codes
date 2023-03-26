import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from threeD_model_final import SpatialSoftmax3D, tile2openpose_conv3d
from model_AutoEncoder import tile2openpose_conv3d_ae
# from model_triplet_branch1 import tile2openpose_conv3d_ae
from model_SetNet_upgrade import tile2pose_se
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
parser.add_argument('--batch_size', type=int, default=64, help='Batch size,128')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--window', type=int, default=10, help='window around the time step')
parser.add_argument('--subsample', type=int, default=1, help='subsample tile res')
parser.add_argument('--linkLoss', type=bool, default=True, help='use link loss')
parser.add_argument('--epoch', type=int, default=100, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--ckpt', type=str, default='cp0_3', help='loaded ckpt file')
parser.add_argument('--eval', type=bool, default=False, help='Set true if eval time')
parser.add_argument('--test_dir', type=str, default='Test/', help='test data path')
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
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).cuda()
    heatmap = torch.where(heatmap < threshold, z, heatmap)
    return heatmap


def check_link(min, max, keypoint):
    # print (torch.max(max), torch.min(min))
    BODY_25_pairs = np.array([
    [1, 8], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [8, 12],
    [12, 13], [13, 14], [1, 0], [14, 15], [15, 16], [14, 17], [11, 18], [18, 19], [11, 20]])
    # o = torch.ones(keypoint.shape[0], keypoint.shape[1], keypoint.shape[2]).cuda()
    # keypoint = torch.where(torch.isnan(keypoint), o, keypoint)
    keypoint_output = torch.ones(keypoint.shape[0], 20).cuda()

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
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # #####

if args.linkLoss:
    link_min = pickle.load(open(args.exp_dir + '../link_min.p', "rb"))
    link_max = pickle.load(open(args.exp_dir + '../link_max.p', "rb"))

    link_min = torch.tensor(link_min, dtype=torch.float).cuda()
    link_max = torch.tensor(link_max, dtype=torch.float).cuda()

if not args.eval:
    train_path = '/home/jovyan/input/dataset/Train/'
    with open('/home/jovyan/input/dataset/train_shuffle.txt', 'r') as f:
        shuffle_train_path = f.readlines()
    mask = []
    train_dataset = sample_data(train_path, shuffle_train_path, args.window, mask, args.subsample)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(len(train_dataset))

    val_path = '/home/jovyan/input/dataset/Val/'
    mask = []
    val_dataset = ValSampleData(val_path, args.window, mask, args.subsample)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(len(val_dataset))

if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    model = tile2openpose_conv3d_ae()  # model
    model = torch.nn.DataParallel(model.cuda(), device_ids=[0])
    # model = tile2openpose_conv3d()
    # model = torch.nn.DataParallel(model.cuda(), device_ids=device)
    softmax = SpatialSoftmax3D(20, 20, 18, 21)

    softmax.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=3, verbose=True)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    criterion = nn.MSELoss()
    triplet_loss = nn.TripletMarginLoss(margin=4)
    triplet_loss_k = nn.TripletMarginLoss(margin=0.05)

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

    train_loss_list = np.zeros(1)
    val_loss_list = np.zeros(1)
    best_val_loss = np.inf
    best_dis = np.inf
    if args.train_continue:
        best_val_loss = 4   # ######
        best_dis = 415

    for epoch in range(epochs+1, args.epoch):
        train_loss = []
        val_loss = []
        print('here')
        keypoint_GT_t = np.empty((1, 21, 3))
        keypoint_pred_t = np.empty((1, 21, 3))
        
        bar = ProgressBar(max_value=len(train_dataloader))
        for i_batch, sample_batched in bar(enumerate(train_dataloader, 0)):
            model.train(True)
            tactile = torch.as_tensor(sample_batched[0], dtype=torch.float).cuda()
            heatmap = torch.as_tensor(sample_batched[1], dtype=torch.float).cuda()
            keypoint = torch.as_tensor(sample_batched[2], dtype=torch.float).cuda()
            n_tactile = torch.as_tensor(sample_batched[3], dtype=torch.float).cuda()
            with torch.set_grad_enabled(True):
                heatmap_out, tactile_out = model(tactile, device)
                heatmap_out = heatmap_out.reshape(-1, 21, 20, 20, 18)
                heatmap_transform = remove_small(heatmap_out.transpose(2, 3), 1e-2)
                keypoint_out, heatmap_out2 = softmax(heatmap_transform * 10)
                n_heatmap_out, n_tactile_out = model(n_tactile, device)
                n_heatmap_out = n_heatmap_out.reshape(-1, 21, 20, 20, 18)
                n_heatmap_transform = remove_small(n_heatmap_out.transpose(2, 3), 1e-2)
                n_keypoint_out, n_heatmap_out2 = softmax(n_heatmap_transform * 10)
            loss_heatmap = torch.mean((heatmap_transform - heatmap)**2 * (heatmap + 0.5) * 2) * 1000
            loss_keypoint = criterion(keypoint_out, keypoint)
            loss_tactile = criterion(tactile_out, tactile) + criterion(n_tactile_out, n_tactile)
            loss_tactile10 = criterion(tactile_10_f, tactile[:, 9:10, :, :]) + criterion(tactile_10_b, tactile[:, 9:10, :, :])
            triplet_keypoint_loss = triplet_loss(keypoint, keypoint_out, n_keypoint_out)
            
            keypoint_GT_t = np.append(keypoint_GT_t, keypoint.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
            keypoint_pred_t = np.append(keypoint_pred_t, keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
            
            a = torch.randint(0, 21, (1, 1))
            p = a+1 if a <= 12 else a-1
            n = a+8 if a <= 12 else a-8
            X_dis = triplet_loss_k(keypoint[:, a, 0], keypoint[:, p, 0], keypoint[:, n, 0])
            X_flag = 0 if X_dis == 0 else 1
            Y_dis = triplet_loss_k(keypoint[:, a, 1], keypoint[:, p, 1], keypoint[:, n, 1])
            Y_flag = 0 if Y_dis == 0 else 1
            Z_dis = triplet_loss_k(keypoint[:, a, 2], keypoint[:, p, 2], keypoint[:, n, 2])
            Z_flag = 0 if Z_dis == 0 else 1
            dis_total = X_flag*torch.exp(X_dis)+Y_flag*torch.exp(Y_dis)+Z_flag*torch.exp(Z_dis)

            W_x = 0 if X_dis == 0 else torch.exp(X_dis)/dis_total
            W_y = 0 if Y_dis == 0 else torch.exp(Y_dis)/dis_total
            W_z = 0 if Z_dis == 0 else torch.exp(Z_dis)/dis_total
          
            X_dis = triplet_loss_k(keypoint_out[:, a, 0], keypoint_out[:, p, 0], keypoint_out[:, n, 0])
            Y_dis = triplet_loss_k(keypoint_out[:, a, 1], keypoint_out[:, p, 1], keypoint_out[:, n, 1])
            Z_dis = triplet_loss_k(keypoint_out[:, a, 2], keypoint_out[:, p, 2], keypoint_out[:, n, 2])

            Loss = W_x*X_dis + W_y*Y_dis + W_z*Z_dis
            
            
            if args.linkLoss:
                loss_link = torch.mean(check_link(link_min, link_max, keypoint_out)) * 10
                loss = loss_link + loss_heatmap + loss_tactile + triplet_keypoint_loss + Loss
            else:
                loss = loss_heatmap + loss_tactile + triplet_keypoint_loss + Loss
            if i_batch % 20 == 0:
                print('loss, triplet', epoch, round(loss.item(), 5), round(triplet_keypoint_loss.item(), 5),
                      round(loss_tactile.item(), 5), optimizer.state_dict()['param_groups'][0]['lr'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.data.item())

            if i_batch % 378 == 0 and i_batch != 0:  # #######

                model.eval()
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

                bar = ProgressBar(max_value=len(val_dataloader))

                c = 0
                for v_i_batch, v_sample_batched in bar(enumerate(val_dataloader, 0)):
                    v_tactile = torch.as_tensor(v_sample_batched[0], dtype=torch.float).cuda()
                    v_heatmap = torch.as_tensor(v_sample_batched[1], dtype=torch.float).cuda()
                    v_keypoint = torch.as_tensor(v_sample_batched[2], dtype=torch.float).cuda()

                    with torch.set_grad_enabled(False):
                        v_heatmap_out, v_tactile_out = model(v_tactile, device)
                        v_heatmap_out = v_heatmap_out.reshape(-1, 21, 20, 20, 18)
                        v_heatmap_transform = remove_small(v_heatmap_out.transpose(2, 3), 1e-2)
                        v_keypoint_out, v_heatmap_out2 = softmax(v_heatmap_transform)

                    v_loss_link = torch.mean(check_link(link_min, link_max, v_keypoint_out))
                    v_loss_heatmap = torch.mean((v_heatmap_transform - v_heatmap) ** 2 * (v_heatmap + 0.5) * 2) * 1000
                    v_heatmap_out = v_heatmap_transform
                    v_loss = v_loss_heatmap + v_loss_link

                    if args.exp_L2:
                        keypoint_GT_log = np.append(keypoint_GT_log, v_keypoint.cpu().data.numpy().reshape(-1, 21, 3),
                                                    axis=0)
                        keypoint_pred_log = np.append(keypoint_pred_log,
                                                      v_keypoint_out.cpu().data.numpy().reshape(-1, 21, 3), axis=0)
                # output L2 distance #
                dis = get_keypoint_spatial_dis(keypoint_GT_log[1:, :, :], keypoint_pred_log[1:, :, :])
                pickle.dump(dis, open(args.exp_dir + 'predictions/L2/' + args.ckpt + '_dis.p', "wb"))
                mean_dis = np.mean(np.mean(np.mean(abs(dis), axis=0), axis=0), axis=0)
                print('mean_dis', mean_dis)
                scheduler.step(mean_dis)
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'loss': loss, },
                           args.exp_dir + 'ckpts1/' + 'cp' + str(epoch) + '_' + str(int(i_batch/378)) + 
                           '_' + str(round(mean_dis, 2)) + '.path.tar')  # ####
                
                if mean_dis < best_dis:
                    print("new_best_keypoint_l2:", mean_dis)
                    best_dis = mean_dis
                    torch.save({
                        'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': best_val_loss, }, args.exp_dir + 'ckpts1/'
                         + 'epoch_' + str(epoch) + '_' + str(int(i_batch/378)) + '_dis_'
                         + str(round(best_dis, 2)) + '.path.tar')  # ####
                dis_t = get_keypoint_spatial_dis(keypoint_GT_t[1:, :, :], keypoint_pred_t[1:, :, :])
                mean_dis_t = np.mean(np.mean(np.mean(abs(dis_t), axis=0), axis=0), axis=0)
                print('train_dis', mean_dis_t)
