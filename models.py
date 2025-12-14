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
#starting channel size of 2**#
start_channels = 4

class Unet2D(nn.Module):
    def __init__(self):
        super().__init__()

        #convolution blocks with batch norm and dropout

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels), 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(2**(start_channels), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+1), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(2**(start_channels+1), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+2), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(2**(start_channels+2), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+3), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(2**(start_channels+3), 2**(start_channels+4), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+4)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+4), 2**(start_channels+4), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+4)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(2**(start_channels+4), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+3), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(2**(start_channels+3), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+2), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2d(2**(start_channels+2), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels+1), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2d(2**(start_channels+1), 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(2**(start_channels), 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm2d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        #last conv, pooling and upconvs
        self.conv10 = nn.Conv2d(2**(start_channels), 1, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.upconv6 = nn.ConvTranspose2d(2**(start_channels+4), 2**(start_channels+3), kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv7 = nn.ConvTranspose2d(2**(start_channels+3), 2**(start_channels+2), kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv8 = nn.ConvTranspose2d(2**(start_channels+2), 2**(start_channels+1), kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv9 = nn.ConvTranspose2d(2**(start_channels+1), 2**(start_channels), kernel_size=upconv_kernel_size, stride=upconv_stride)


    def forward(self, x):

        #encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)  
        c3 = self.conv3(p2)
        p3 = self.pool(c3)  
        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        #bottleneck
        c5 = self.conv5(p4)

        #decoder
        u6 = torch.cat((self.upconv6(c5), c4), dim=1)
        c6 = self.conv6(u6)
        u7 = torch.cat((self.upconv7(c6), c3), dim=1)
        c7 = self.conv7(u7)
        u8 = torch.cat((self.upconv8(c7), c2), dim=1)
        c8 = self.conv8(u8)
        u9 = torch.cat((self.upconv9(c8), c1), dim=1)
        c9 = self.conv9(u9)

        output = torch.sigmoid(self.conv10(c9))
        
        return output
    
    
class Unet3D(nn.Module):
    def __init__(self):
        super().__init__()

        #convolution blocks with batch norm and dropout

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels), 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(2**(start_channels), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+1), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(2**(start_channels+1), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+2), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(2**(start_channels+2), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+3), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(2**(start_channels+3), 2**(start_channels+4), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+4)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+4), 2**(start_channels+4), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+4)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv6 = nn.Sequential(
            nn.Conv3d(2**(start_channels+4), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+3), 2**(start_channels+3), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+3)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(2**(start_channels+3), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+2), 2**(start_channels+2), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+2)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv8 = nn.Sequential(
            nn.Conv3d(2**(start_channels+2), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels+1), 2**(start_channels+1), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels+1)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )

        self.conv9 = nn.Sequential(
            nn.Conv3d(2**(start_channels+1), 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(2**(start_channels), 2**(start_channels), kernel_size=conv_kernel_size, padding=conv_padding),
            nn.BatchNorm3d(2**(start_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
        )
        
        #last conv, pooling and upconvs
        self.conv10 = nn.Conv3d(2**(start_channels), 1, kernel_size=1)

        self.pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride)

        self.upconv6 = nn.ConvTranspose3d(2**(start_channels+4), 2**(start_channels+3), kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv7 = nn.ConvTranspose3d(2**(start_channels+3), 2**(start_channels+2), kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv8 = nn.ConvTranspose3d(2**(start_channels+2), 2**(start_channels+1), kernel_size=upconv_kernel_size, stride=upconv_stride)
        self.upconv9 = nn.ConvTranspose3d(2**(start_channels+1), 2**(start_channels), kernel_size=upconv_kernel_size, stride=upconv_stride)


    def forward(self, x):

        #encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)
        c2 = self.conv2(p1)
        p2 = self.pool(c2)  
        c3 = self.conv3(p2)
        p3 = self.pool(c3)  
        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        #bottleneck
        c5 = self.conv5(p4)

        #decoder
        u6 = torch.cat((self.upconv6(c5), c4), dim=1)
        c6 = self.conv6(u6)
        u7 = torch.cat((self.upconv7(c6), c3), dim=1)
        c7 = self.conv7(u7)
        u8 = torch.cat((self.upconv8(c7), c2), dim=1)
        c8 = self.conv8(u8)
        u9 = torch.cat((self.upconv9(c8), c1), dim=1)
        c9 = self.conv9(u9)

        output = torch.sigmoid(self.conv10(c9))
        
        return output