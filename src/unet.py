import torch 
import torch.nn as nn

class ConvBlock(nn.Module):
    """ Convolution block """
    def __init__(self,in_c,out_c,ksize=3):
        super().__init__()

        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=ksize,stride=1,padding=1), # (in_c,H,W) -> (out_c,H,W)
                nn.ReLU(),
                nn.Conv2d(in_channels=out_c,out_channels=out_c,kernel_size=ksize,stride=1,padding=1), # (out_c,H,W) -> (out_c,H,W)
                nn.ReLU(),
                )

    def forward(self,inputs):
        return self.conv(inputs)

class EncBlock(nn.Module):
    """ Encoder Block - down sampling """
    def __init__(self,in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c,out_c) # (in_c,H,W) -> (out_c,H,W) 
        self.down = nn.MaxPool2d((2,2)) # (out_c,H,W) -> (out_c,H/2,W/2)

    def forward(self,inputs):
        x = self.conv(inputs)
        p = self.down(x)

        return x, p

class DecBlock(nn.Module):
    """ Decoder Block - up sampling"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels = in_c, out_channels = out_c, kernel_size=2, stride=2, padding=0) # (in_c,H,W) -> (out_c, 2H, 2W)
        self.conv = ConvBlock(2*out_c,out_c)

    def forward(self,inputs,to_concat):
        x = self.upsample(inputs)
        x = torch.cat([x,to_concat],axis=0)
        x = self.conv(x)

        return x

class UNET(nn.Module): 
    # lattice deformation params: (dim_g, Lx, Ly) 
    def __init__(self, dim_C):
        super().__init__()

        n = dim_C
        dim_g = n**2 + 2*n # CP(n) -> su(n+1)

        self.enc1 = EncBlock(in_c = 2*(n+1), out_c = 64) # (2*(n+1), Lx, Ly)-> (64,Lx/2,Ly/2)
        self.enc2 = EncBlock(in_c = 64, out_c = 128)   # -> (128,Lx/4,Ly/4)
        self.enc3 = EncBlock(in_c = 128, out_c = 256)  # -> (256, Lx/8,Ly/8)

        self.bottle_neck = ConvBlock(in_c = 256, out_c=512) # bottle neck -> (1024,Lx/8,Ly/8)

        self.dec1 = DecBlock(in_c = 512,out_c=256)  # -> (256,Lx/4,Ly/4)
        self.dec2 = DecBlock(in_c = 256,out_c=128)  # -> (128,Lx/2,Ly/2)
        self.dec3 = DecBlock(in_c = 128,out_c=64)   # -> (64,Lx,Ly)

        self.output = nn.Conv2d(in_channels=64,out_channels=dim_g,kernel_size=3,stride=1,padding=1) #(64,Lx,Ly) -> (dim_g,Lx,Ly)

        self.set_inf_rnd_weights() # set kernel weights once (<< 1, rndn)

    def forward(self, mask):
        # down sampling
        lvl1, down1 = self.enc1(mask)
        lvl2, down2 = self.enc2(down1)
        lvl3, down3 = self.enc3(down2)

        # bottle neck
        bn = self.bottle_neck(down3)

        # upsampling
        up1 = self.dec1(bn,lvl3)
        up2 = self.dec2(up1,lvl2)
        up3 = self.dec3(up2,lvl1)

        output = self.output(up3)

        return output

    def set_inf_rnd_weights(self):
        """ Sets all kernel weights to zero """

        with torch.no_grad():
            for param in self.parameters():
                param.copy_(torch.randn_like(param) * 1e-8)

        
