import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICE'] = '3'

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv2d(ipts)
        return outs


class GLConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=3, stride=1, padding=1, bias=False, **kwargs):
        super(GLConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.global_conv2d = BasicConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv2d = BasicConv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(2**halving, 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4, 2**halving, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        gob_feat = self.global_conv2d(x)
        if self.halving == 0:
            lcl_feat = self.local_conv2d(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 2)
            lcl_feat1 = lcl_feat
            # todo Attention  
            y = [self.avg_pool(_) for _ in lcl_feat1]
            # y = torch.tensor(np.array([item.cpu().detach().numpy() for item in y]))
            print(type(y))
            y = torch.tensor([item.cpu().detach().numpy() for item in y])
            y = y.transpose(0, 1).contiguous().view(y.size(1), y.size(0))
            y = self.fc(y).contiguous().view(y.size(0), y.size(1), 1, 1, 1)  # tensor
            lcl_feat = [self.local_conv2d(_) for _ in lcl_feat]  # list
            for i in range(2**self.halving):
                lcl_feat[i] *= y[:, i, :, :, :].expand_as(lcl_feat[i])
            lcl_feat = torch.cat(lcl_feat, 2)

        if not self.fm_sign:
            feat = F.leaky_relu(gob_feat) + F.leaky_relu(lcl_feat)
        else:
            feat = F.leaky_relu(torch.cat([gob_feat, lcl_feat], dim=2))
        return feat


class tile2pose_GL(nn.Module):
    def __init__(self, *args, **kargs):
        super(tile2pose_GL, self).__init__(*args, **kargs)
    
        self.conv_0 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.GLConvA0 = GLConv(32, 32, halving=3, fm_sign=False, kernel_size=3, stride=1, padding=1)
        
        self.conv_1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2)) # 48 * 48

        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)) # 24 * 24

        self.conv_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))

        self.conv_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5,5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))

        self.conv_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2)) # 10 * 10

        self.convTrans_0 = nn.Sequential(
            nn.Conv3d(1025, 1025, kernel_size=(3,3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1025))

        self.convTrans_1 = nn.Sequential(
            nn.Conv3d(1025, 512, kernel_size=(3,3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(512))

        self.convTrans_00 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=(2,2,2),stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256))

        self.convTrans_2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=(3,3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128))

        self.convTrans_3 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3,3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(64))

        self.convTrans_4 =  nn.Sequential(
            nn.Conv3d(64, 21, kernel_size=(3,3,3),padding=1),
            nn.Sigmoid())
            
        # output 1024x20x20
        self.convTranspose_0 = nn.Sequential(
            nn.ConvTranspose2d(1024, 1024, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # ####output 512x24x24
        self.convTranspose_1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=(5,5), padding=0),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))
        
        # output 256x24x24
        self.convTranspose_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256))
        
        # output 128x48x48
        self.convTranspose_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # output 64x48x48
        self.convTranspose_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64))
        
        # output 32x96x96
        self.convTranspose_5 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        
        # output 20x96x96
        self.convTranspose_6 = nn.Sequential(
            nn.ConvTranspose2d(32, 20, kernel_size=(3,3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(20))    

    def forward(self, input, device):
        output = self.conv_0(input)
        output = self.GLConvA0(output)
        output = self.conv_1(output)
        output = self.conv_2(output)
        output = self.conv_3(output)
        output = self.conv_4(output)
        output = self.conv_5(output)
        output = self.conv_6(output)
        output1 = output
        output1 = self.convTranspose_0(output1)# output 1024x20x20
        output1 = self.convTranspose_1(output1)# ####output 512x24x24
        output1 = self.convTranspose_2(output1)# output 256x24x24
        output1 = self.convTranspose_3(output1)# output 128x48x48
        output1 = self.convTranspose_4(output1)# output 64x48x48
        output1 = self.convTranspose_5(output1)# output 32x96x96
        output1 = self.convTranspose_6(output1)# output 20x96x96
        output = output.reshape(output.shape[0],output.shape[1],output.shape[2],output.shape[3],1)
        output = output.repeat(1,1,1,1,9)

        layer = torch.zeros(output.shape[0], 1, output.shape[2], output.shape[3], output.shape[4]).to(device)
        for i in range(layer.shape[4]):
            layer[:,:,:,:,i] = i
        layer = layer/(layer.shape[4]-1)
        output = torch.cat((output,layer), axis=1)

        # print (output.shape)

        output = self.convTrans_0(output)
        output = self.convTrans_1(output)
        output = self.convTrans_00(output)
        output = self.convTrans_2(output)
        output = self.convTrans_3(output)
        output = self.convTrans_4(output)

        # print (output.shape)

        return output, output1



if __name__ == "__main__":
    device = 'cpu'
    model = tile2pose_GL()
    tactile = np.random.randn(16, 20, 96, 96)
    tactile = tactile.astype(np.float32)
    tactile = torch.as_tensor(tactile, dtype=torch.float)
    heatmap, tactile_out = model(tactile, device)
    print(heatmap.shape, tactile_out.shape)

