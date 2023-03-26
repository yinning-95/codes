import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils_func import softmax

def softmax(data):
    for i in range(data.shape[0]):
        f = data[i,:].reshape (data.shape[1])
        data[i,:] = torch.exp(f) / torch.sum(torch.exp(f))
    return data


class SpatialSoftmax3D(torch.nn.Module):
    def __init__(self, height, width, depth, channel, lim=[0., 1., 0., 1., 0., 1.], temperature=None, data_format='NCHWD'):
        super(SpatialSoftmax3D, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.depth = depth
        self.channel = channel
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.
        pos_y, pos_x, pos_z = np.meshgrid(
            np.linspace(lim[0], lim[1], self.width),
            np.linspace(lim[2], lim[3], self.height),
            np.linspace(lim[4], lim[5], self.depth))
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width * self.depth)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width * self.depth)).float()
        pos_z = torch.from_numpy(pos_z.reshape(self.height * self.width * self.depth)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)
        self.register_buffer('pos_z', pos_z)
    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...
        if self.data_format == 'NHWDC':
            feature = feature.transpose(1, 4).tranpose(2, 4).tranpose(3,4).reshape(-1, self.height * self.width * self.depth)
        else:
            feature = feature.reshape(-1, self.height * self.width * self.depth)
        softmax_attention = feature
        # softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        heatmap = softmax_attention.reshape(-1, self.channel, self.height, self.width, self.depth)

        eps = 1e-6
        expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_z = torch.sum(self.pos_z * softmax_attention, dim=1, keepdim=True)/(torch.sum(softmax_attention, dim=1, keepdim=True) + eps)
        expected_xyz = torch.cat([expected_x, expected_y, expected_z], 1)
        feature_keypoints = expected_xyz.reshape(-1, self.channel, 3)
        return feature_keypoints, heatmap

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): # x [batch_size, C, H, W]
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
        
class UseAttentionModel(nn.Module): # 这里可以随便定义自己的模型
    def __init__(self, H):
        super(UseAttentionModel, self).__init__()
        self.channel_attention = ChannelAttention(H)

    def forward(self, x):  # 反向传播
        attention_value = self.channel_attention(x)
        out = x.mul(attention_value) # 得到借助注意力机制后的输出
        return out

class tile2openpose_conv3d_ae(nn.Module):
    def __init__(self):
        super(tile2openpose_conv3d_ae, self).__init__()   #tactile 96*96
        self.conv_0 = nn.Sequential(
            nn.Conv2d(20, 32, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

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

        # Encoder 1
        self.Econv_1_0 = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.Econv_1_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))

        self.Econv_1_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.Econv_1_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))

        self.Econv_1_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))

        self.Econv_1_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))

        self.Econv_1_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2))

        # Encoder 2
        self.Econv_2_0 = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))

        self.Econv_2_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))

        self.Econv_2_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.Econv_2_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2))

        self.Econv_2_4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(512))

        self.Econv_2_5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(5, 5)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024))

        self.Econv_2_6 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(kernel_size=2))

        #  Decoder 1
        self.Dconv1_0 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=(5, 5), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )

        self.Dconv1_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.Dconv1_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.Dconv1_3 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.Dconv1_4 = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1)
        )

        #  Decoder 2
        self.Dconv2_0 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=(5, 5), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(256)
        )

        self.Dconv2_1 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.Dconv2_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 16, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.Dconv2_3 = nn.Sequential(
            nn.ConvTranspose2d(16, 4, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(4),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.Dconv2_4 = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(1)
        )


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
        # self.convTrans_4 =  nn.Conv3d(64, 21, kernel_size=(3,3,3),padding=1)


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
        
        self.conv1x1 = nn.Conv2d(3072, 1024, kernel_size=(1, 1), padding=0)

    def forward(self, input, device):
        tactile_f = input[:, 0:10, :, :]
        tactile_b = input[:, 11:20, :, :]

        tactile_f = self.Econv_1_0(tactile_f)
        tactile_f = self.Econv_1_1(tactile_f)
        tactile_f = self.Econv_1_2(tactile_f)
        tactile_f = self.Econv_1_3(tactile_f)
        tactile_f = self.Econv_1_4(tactile_f)
        tactile_f = self.Econv_1_5(tactile_f)
        tactile_f = self.Econv_1_6(tactile_f)

        tactile_b = self.Econv_2_0(tactile_b)
        tactile_b = self.Econv_2_1(tactile_b)
        tactile_b = self.Econv_2_2(tactile_b)
        tactile_b = self.Econv_2_3(tactile_b)
        tactile_b = self.Econv_2_4(tactile_b)
        tactile_b = self.Econv_2_5(tactile_b)
        tactile_b = self.Econv_2_6(tactile_b)

        heatmap_out = self.conv_0(input)
        heatmap_out = self.conv_1(heatmap_out)
        heatmap_out = self.conv_2(heatmap_out)
        heatmap_out = self.conv_3(heatmap_out)
        heatmap_out = self.conv_4(heatmap_out)
        heatmap_out = self.conv_5(heatmap_out)
        heatmap_out = self.conv_6(heatmap_out)
        tactile_20 = heatmap_out
        tactile_20 = self.convTranspose_0(tactile_20)  # output 1024x20x20
        tactile_20 = self.convTranspose_1(tactile_20)  # ####output 512x24x24
        tactile_20 = self.convTranspose_2(tactile_20)  # output 256x24x24
        tactile_20 = self.convTranspose_3(tactile_20)  # output 128x48x48
        tactile_20 = self.convTranspose_4(tactile_20)  # output 64x48x48
        tactile_20 = self.convTranspose_5(tactile_20)  # output 32x96x96
        tactile_20 = self.convTranspose_6(tactile_20)  # output 20x96x96
        heatmap_out = torch.cat((heatmap_out, tactile_f, tactile_b), axis=1)
        
        # todo channel Attention
        attention = UseAttentionModel(3072).to(device)
        heatmap_out = attention(heatmap_out)
        heatmap_out = self.conv1x1(heatmap_out)
        
        tactile_f = self.Dconv1_0(tactile_f)
        tactile_f = self.Dconv1_1(tactile_f)
        tactile_f = self.Dconv1_2(tactile_f)
        tactile_f = self.Dconv1_3(tactile_f)
        tactile_f = self.Dconv1_4(tactile_f)

        tactile_b = self.Dconv2_0(tactile_b)
        tactile_b = self.Dconv2_1(tactile_b)
        tactile_b = self.Dconv2_2(tactile_b)
        tactile_b = self.Dconv2_3(tactile_b)
        tactile_b = self.Dconv2_4(tactile_b)

        heatmap_out = heatmap_out.reshape(heatmap_out.shape[0],heatmap_out.shape[1],heatmap_out.shape[2],heatmap_out.shape[3],1)
        heatmap_out = heatmap_out.repeat(1,1,1,1,9)

        layer = torch.zeros(heatmap_out.shape[0], 1, heatmap_out.shape[2], heatmap_out.shape[3], heatmap_out.shape[4]).to(device)
        for i in range(layer.shape[4]):
            layer[:,:,:,:,i] = i
        layer = layer/(layer.shape[4]-1)
        heatmap_out = torch.cat((heatmap_out, layer), axis=1)

        # print (output.shape)

        heatmap_out = self.convTrans_0(heatmap_out)
        heatmap_out = self.convTrans_1(heatmap_out)
        heatmap_out = self.convTrans_00(heatmap_out)
        heatmap_out = self.convTrans_2(heatmap_out)
        heatmap_out = self.convTrans_3(heatmap_out)
        heatmap_out = self.convTrans_4(heatmap_out)

        # print (output.shape)

        return heatmap_out, tactile_20, tactile_f, tactile_b


'''
criterion = nn.MSELoss()
device = 'cpu'
model = tile2openpose_conv3d_ae(10)
softmax = SpatialSoftmax3D(20, 20, 18, 21)
model.to(device)
softmax.to(device)
input_tactile = torch.rand(32, 20, 96, 96)
out_heatmap, output_tactile = model(input_tactile, device)
print(output_tactile.shape)
print(criterion(output_tactile, input_tactile))
'''