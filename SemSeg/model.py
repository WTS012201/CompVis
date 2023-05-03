import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class Convblock(nn.Module):
      def __init__(self,in_channel,out_channel,kernal=3,stride=1,padding=1):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channel
            ,out_channel,kernal,stride,padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel,out_channel,kernal),
            nn.ReLU(inplace=True),
        )

      def forward(self,x):
        x = self.convblock(x)
        return x

class UNet(nn.Module):
    
    def __init__(self,in_channel ,retain=True):
        super().__init__()

        self.conv1 = Convblock(in_channel ,32)
        self.conv2 = Convblock(32,64)
        self.conv3 = Convblock(64,128)
        self.conv4 = Convblock(128,256)
        self.dropout1 = nn.Dropout(p=0.2)
        self.neck = nn.Conv2d(256,512,3,1)
        self.upconv4 = nn.ConvTranspose2d(512,256,3,2,0,1)
        self.dconv4 = Convblock(512,256)
        self.upconv3 = nn.ConvTranspose2d(256,128,3,2,0,1)
        self.dconv3 = Convblock(256,128)
        self.upconv2 = nn.ConvTranspose2d(128,64,3,2,0,1)
        self.dconv2 = Convblock(128,64)
        self.upconv1 = nn.ConvTranspose2d(64,32,3,2,0,1)
        self.dconv1 = Convblock(64,32)
        self.dropout2 = nn.Dropout(p=0.2)
        self.out = nn.Conv2d(32,3,1,1)

        self.retain = retain
        
    def forward(self,x):
        
        # Encoder
        conv1 = self.conv1(x)
        pool1 = F.max_pool2d(conv1,kernel_size=2,stride=2)
        conv2 = self.conv2(pool1)
        pool2 = F.max_pool2d(conv2,kernel_size=2,stride=2)
        conv3 = self.conv3(pool2)
        pool3 = F.max_pool2d(conv3,kernel_size=2,stride=2)
        conv4 = self.conv4(pool3)
        pool4 = F.max_pool2d(conv4,kernel_size=2,stride=2)

        # BottleNeck
        dropout1 = self.dropout1(pool4)
        neck = self.neck(dropout1)
        
        # Decoder
        upconv4 = self.upconv4(neck)
        croped = self.crop(conv4,upconv4)
        dconv4 = self.dconv4(torch.cat([upconv4,croped],1))
        upconv3 = self.upconv3(dconv4)
        croped = self.crop(conv3,upconv3)
        dconv3 = self.dconv3(torch.cat([upconv3,croped],1))
        upconv2 = self.upconv2(dconv3)
        croped = self.crop(conv2,upconv2)
        dconv2 = self.dconv2(torch.cat([upconv2,croped],1))
        upconv1 = self.upconv1(dconv2)
        croped = self.crop(conv1,upconv1)
        dconv1 = self.dconv1(torch.cat([upconv1,croped],1))

        # Output
        dropout2 = self.dropout2(dconv1)
        out = self.out(dropout2)
        
        if self.retain == True:
            out = F.interpolate(out,list(x.shape)[2:])

        return out
    
    def crop(self, input_tensor, target_tensor):
        _,_,H,W = target_tensor.shape
        return T.CenterCrop([H,W])(input_tensor)