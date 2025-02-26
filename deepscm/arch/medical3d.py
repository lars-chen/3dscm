import torch
import torch.nn as nn
import torch.utils.data
import numpy as np



def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm3d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")
        
class SE(nn.Module):
    """
    Squeeze-and-Excitation 
    """
    def __init__(self, channel, reduction_ratio=4):
        super(SE, self).__init__()
        
        ### Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        ### Fully Connected Multi-Layer Perceptron (FC-MLP)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class ds_Conv3d(nn.Module):
    """
    Depthwise-Seperable 3D Convolution
    """
    def __init__(self, nin, nout, kernel_size=3, stride=1, padding=1, kernels_per_layer=1):
        super(ds_Conv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, stride=stride, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv3d(channel_in, (channel_out // 2) + channel_out, kernel_size, 2, kernel_size // 2)
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv3d(channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2)

        self.act_fnc = nn.SiLU()
        
        self.SE = SE(channel_out)
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        x = self.SE(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=5, scale_factor=2, norm_type="bn"): # kernel size 3 TODO
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = ds_Conv3d(channel_in, (channel_in // 2) + channel_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = ds_Conv3d(channel_in // 2, channel_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.SiLU()
        
        self.SE = SE(channel_out)

        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, :self.channel_out]
        x = x_cat[:, self.channel_out:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        x = self.SE(x)
        
        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = channel_in // 2 if channel_in == channel_out else (channel_in // 2) + channel_out
        
        self.conv1 = ds_Conv3d(channel_in, first_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        #self.conv1 = nn.Conv3d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = ds_Conv3d(channel_in // 2, channel_out, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        #self.conv2 = nn.Conv3d(channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2)
        self.act_fnc = nn.SiLU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2
        
        self.SE = SE(channel_out)


    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, :self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk:]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)
        x = self.SE(x)

        return x + skip


class Encoder3D(nn.Module):
    """
    Encoder block
    """

    def __init__(self, channel_in, ch, blocks, latent_dim, latent_channels, image_shape, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Encoder3D, self).__init__()
        self.ch = ch
        self.image_shape = image_shape
        self.conv_in = nn.Conv3d(channel_in, blocks[0] * self.ch, 3, 1, 1)
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]
        self.act_fnc = nn.ELU()
        
        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional non down-sampling block before down-sampling
                self.layer_blocks.append(ResBlock(w_in * self.ch, w_in * self.ch, norm_type=norm_type))

            self.layer_blocks.append(ResDown(w_in * ch, w_out * ch, norm_type=norm_type))

        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type))
        
        self.final_conv = nn.Conv3d(widths_out[-1] * ch, self.latent_channels, 1, 1) # 64 --> latent_channels
        
        self.intermediate_shape = np.array(self.image_shape) // (2**len(blocks))
        self.intermediate_shape[0] = self.latent_channels
        
        self.fc = nn.Sequential(
            nn.Linear(int(np.prod(self.intermediate_shape)), self.latent_dim), 
            nn.BatchNorm1d(self.latent_dim), 
            nn.LeakyReLU(0.1, inplace=True)
        )
        

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) 
        return mu + eps * std

    def forward(self, x):
        x = self.conv_in(x.type(torch.float32))

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)
        
        x = self.final_conv(x).view(-1, self.intermediate_shape.prod())
        x = self.fc(x)
        

        return x


class Decoder3D(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(self, channels, ch, blocks, latent_dim, context_dim, latent_channels, image_shape, num_res_blocks=1, norm_type="bn",
                 deep_model=False):
        super(Decoder3D, self).__init__()

        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]
        self.image_shape = image_shape
        self.latent_channels = latent_channels
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        
        self.intermediate_shape = np.array(self.image_shape ) / (2**len(blocks))
        self.intermediate_shape[0] = latent_channels
        self.ch = ch

        self.fc_in = nn.Sequential(
            nn.Linear(latent_dim+self.context_dim, int(np.prod(self.intermediate_shape))), #TODO jankyyy af
            nn.BatchNorm1d(int(np.prod(self.intermediate_shape))), 
            nn.LeakyReLU(.1, inplace=True)
        )
        self.latent_channels = latent_channels
        self.intermediate_shape = np.array(image_shape)//(2**len(blocks))
        self.intermediate_shape[0] = self.latent_channels

        self.conv_in = ds_Conv3d(latent_channels, widths_in[0] * self.ch, kernel_size=1, stride=1, padding=0)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(ResBlock(widths_in[0] * self.ch, widths_in[0] * ch, norm_type=norm_type))

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * self.ch, w_out * self.ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(ResBlock(w_out * self.ch, w_out * self.ch, norm_type=norm_type))

        self.conv_out = ds_Conv3d(blocks[0] * self.ch, channels, kernel_size=5, stride=1, padding=2)
        self.act_fnc = nn.ELU()

    def forward(self, x):

        x = self.fc_in(x).view(-1, *np.int64(self.intermediate_shape))
        
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        return self.conv_out(x)# torch.tanh(self.conv_out(x)) # TODO: probably not tanh!!! self.conv_out(x) #
