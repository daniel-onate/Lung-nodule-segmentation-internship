import torch
import torch.nn as nn
import torch.nn.functional as F

#variables
conv_kernel_size = 3
conv_padding = 1
upconv_kernel_size = 2
upconv_stride = 2
pool_kernel_size = 2
pool_stride = 2


class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        #encoder
        self.conv1_1 = nn.Conv3d(1, 32, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv1_2 = nn.Conv3d(32, 32, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv2_1 = nn.Conv3d(32, 64, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv2_2 = nn.Conv3d(64, 64, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv3_1 = nn.Conv3d(64, 128, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv3_2 = nn.Conv3d(128, 128, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv4_1 = nn.Conv3d(128, 256, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv4_2 = nn.Conv3d(256, 256, kernel_size=conv_kernel_size, padding=conv_padding)

        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.drop1 = nn.Dropout3d(0.1)
        self.drop2 = nn.Dropout3d(0.1)
        self.drop3 = nn.Dropout3d(0.1)
        self.drop4 = nn.Dropout3d(0.1)
        self.drop5 = nn.Dropout3d(0.1)
        self.drop6 = nn.Dropout3d(0.1)
        self.drop7 = nn.Dropout3d(0.1)
        self.drop8 = nn.Dropout3d(0.1)
        self.drop9 = nn.Dropout3d(0.1)

        #bottleneck
        self.conv5_1 = nn.Conv3d(256, 512, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv5_2 = nn.Conv3d(512, 512, kernel_size=conv_kernel_size, padding=conv_padding)

        #decoder
        self.conv6_1 = nn.Conv3d(512, 256, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv6_2 = nn.Conv3d(256, 256, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv7_1 = nn.Conv3d(256, 128, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv7_2 = nn.Conv3d(128, 128, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv8_1 = nn.Conv3d(128, 64, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv8_2 = nn.Conv3d(64, 64, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv9_1 = nn.Conv3d(64, 32, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv9_2 = nn.Conv3d(32, 32, kernel_size=conv_kernel_size, padding=conv_padding)
        self.conv10 = nn.Conv3d(32, 1, kernel_size=1)

        self.upconv6 = nn.ConvTranspose3d(512, 256, kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv7 = nn.ConvTranspose3d(256, 128, kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv8 = nn.ConvTranspose3d(128, 64, kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv9 = nn.ConvTranspose3d(64, 32, kernel_size=upconv_kernel_size, stride=upconv_stride)


    def forward(self, x):

        #encoder
        c1 = F.relu(self.conv1_1(x))
        c1 = self.drop1(c1)
        c1 = F.relu(self.conv1_2(c1))
        p1 = self.pool(c1)
        c2 = F.relu(self.conv2_1(p1))
        c2 = self.drop2(c2)
        c2 = F.relu(self.conv2_2(c2))
        p2 = self.pool(c2)  
        c3 = F.relu(self.conv3_1(p2))
        c3 = self.drop3(c3)
        c3 = F.relu(self.conv3_2(c3))
        p3 = self.pool(c3)  
        c4 = F.relu(self.conv4_1(p3))
        c4 = self.drop4(c4)
        c4 = F.relu(self.conv4_2(c4))
        p4 = self.pool(c4)

        #bottleneck
        c5 = F.relu(self.conv5_1(p4))
        c5 = self.drop5(c5)
        c5 = F.relu(self.conv5_2(c5))

        #decoder
        u6 = torch.cat((self.upconv6(c5), c4), dim=1)
        c6 = F.relu(self.conv6_1(u6))
        c6 = self.drop6(c6)
        c6 = F.relu(self.conv6_2(c6))
        u7 = torch.cat((self.upconv7(c6), c3), dim=1)
        c7 = F.relu(self.conv7_1(u7))
        c7 = self.drop7(c7)
        c7 = F.relu(self.conv7_2(c7))
        u8 = torch.cat((self.upconv8(c7), c2), dim=1)
        c8 = F.relu(self.conv8_1(u8))
        c8 = self.drop8(c8)
        c8 = F.relu(self.conv8_2(c8))
        u9 = torch.cat((self.upconv9(c8), c1), dim=1)
        c9 = F.relu(self.conv9_1(u9))
        c9 = self.drop9(c9)
        c9 = F.relu(self.conv9_2(c9))
        output = torch.sigmoid(self.conv10(c9))
        
        return output