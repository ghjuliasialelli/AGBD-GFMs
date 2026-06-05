"""
A CNN designed for pixel-wise analysis of Sentinel-2 satellite images.
"XceptionS2" builds on the separable convolution described by Chollet (2017) who proposed the Xception network.
Any kind of down sampling is avoided (no pooling, striding, etc.).

All details about the architecture are described in:
Lang, N., Schindler, K., Wegner, J.D.: Country-wide high-resolution vegetation height mapping with Sentinel-2,
Remote Sensing of Environment, vol. 233 (2019) <https://arxiv.org/abs/1904.13270>
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=True)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, padding_mode='zeros', valid_padding=False):
        super(SeparableConv2d, self).__init__()

        self.valid_padding = valid_padding
        if valid_padding : 
            self.up = nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 0)
            padding = 0

        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation,
                                   groups=in_channels, bias=False)

        self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                   padding=0, padding_mode=padding_mode, dilation=1, groups=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.valid_padding : x = self.up(x)
        x = self.pointwise(x)
        return x

class film(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(film, self).__init__()
        self.layer = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x, b):
        scale_shift = self.layer(b)
        scale, shift = scale_shift[:, :, None, None].chunk(2, dim=1)
        scale = scale + 1
        return scale * x + shift

class PointwiseBlock(nn.Module):

    def __init__(self, in_channels, filters, norm_layer=nn.BatchNorm2d, biome_dim=None):
        super(PointwiseBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters

        self.conv1 = conv1x1(in_channels, filters[0])
        self.film1 = film(biome_dim, filters[0])
        self.bn1 = norm_layer(filters[0])

        self.conv2 = conv1x1(filters[0], filters[1])
        self.film2 = film(biome_dim, filters[1])
        self.bn2 = norm_layer(filters[1])

        self.conv3 = conv1x1(filters[1], filters[2])
        self.film3 = film(biome_dim, filters[2])
        self.bn3 = norm_layer(filters[2])

        self.relu = nn.ReLU(inplace=True)
        self.conv_shortcut = conv1x1(in_channels, filters[2])
        self.film_shortcut = film(biome_dim, filters[2])
        self.bn_shortcut = norm_layer(filters[2])

    def forward(self, x, b = None):
        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)
            if b is not None: shortcut = self.film_shortcut(shortcut, b)

        out = self.conv1(x)
        out = self.bn1(out)
        if b is not None: out = self.film1(out, b)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if b is not None: out = self.film2(out, b)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if b is not None: out = self.film3(out, b)

        out = out + shortcut
        out = self.relu(out)

        return out


class SepConvBlock(nn.Module):

    def __init__(self, in_channels, filters, norm_layer=nn.BatchNorm2d, biome_dim=None, only_entry=False, padding_mode='zeros', valid_padding=False):
        super(SepConvBlock, self).__init__()

        self.in_channels = in_channels
        self.filters = filters
        self.only_entry = only_entry

        self.sepconv1 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3, padding_mode=padding_mode, valid_padding=valid_padding)
        self.bn1 = norm_layer(filters[0])
        if not only_entry: self.film1 = film(biome_dim, filters[0])

        self.sepconv2 = SeparableConv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=3, padding_mode=padding_mode, valid_padding=valid_padding)
        self.bn2 = norm_layer(filters[1])
        if not only_entry: self.film2 = film(biome_dim, filters[1])

        self.relu = nn.ReLU(inplace=False)
        self.conv_shortcut = conv1x1(in_channels, filters[1])
        self.bn_shortcut = norm_layer(filters[1])
        if not only_entry: self.film_shortcut = film(biome_dim, filters[1])

    def forward(self, inputs):

        x, b = inputs

        if self.in_channels == self.filters[-1]:
            # identity shortcut
            shortcut = x
        else:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut)
            if (b is not None) and (not self.only_entry): shortcut = self.film_shortcut(shortcut, b)

        out = self.relu(x)
        out = self.sepconv1(out)
        out = self.bn1(out)
        if (b is not None) and (not self.only_entry): out = self.film1(out, b)

        out = self.relu(out)
        out = self.sepconv2(out)
        out = self.bn2(out)
        if (b is not None) and (not self.only_entry): out = self.film2(out, b)

        out = out + shortcut
        outputs = (out, b)

        return outputs


class XceptionS2_FiLM(nn.Module):

    def __init__(self, in_channels, out_channels=1, num_sepconv_blocks=8, num_sepconv_filters=728, returns="dense",
                long_skip=False, manual_init=False, freeze_features=False, freeze_last_mean=False, biome_dim=None,
                only_entry=False, padding_mode='zeros', patch_size=15, valid_padding=False, sigreg_lambda=0.0):

        super(XceptionS2_FiLM, self).__init__()

        self.freeze_features = freeze_features
        self.freeze_last_mean = freeze_last_mean  # freeze the last linear regression layers (mean)
        self.valid_padding = valid_padding

        self.num_sepconv_blocks = num_sepconv_blocks
        self.num_sepconv_filters = num_sepconv_filters
        self.returns = returns
        self.long_skip = long_skip

        self.entry_block = PointwiseBlock(in_channels=in_channels, filters=[128, 256, num_sepconv_filters], biome_dim=biome_dim)
        self.sepconv_blocks = self._make_sepconv_blocks(biome_dim, only_entry, padding_mode)

        self.predictions = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)
        self.second_moments = conv1x1(in_channels=num_sepconv_filters, out_channels=out_channels)

        if self.returns == "pixel" :
            self.pixelwise = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=patch_size, stride=1, bias=True)
        
        self.return_latents = sigreg_lambda > 0.0

        # initialize parameters
        if manual_init:
            print('Manual weight init with Kaiming Normal')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=1.0)
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') TODO: check if kaiming would be better with ReLU (see torchvision resnet)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)  # gamma
                    nn.init.constant_(m.bias, 0)  # beta

        if self.freeze_features:
            print(
                f'Freezing feature extractor... args.freeze_features={self.freeze_features}'
            )
            # do not train the backbone of the image network
            for param in self.parameters():
                param.requires_grad = False

        # train the last layer(s) of the linear regressor
        if not self.freeze_last_mean:
            print(
                f'Unfreeze last layer (mean regressor)... args.freeze_last_mean={self.freeze_last_mean}'
            )
            for param in self.predictions.parameters():
                param.requires_grad = True

    def _make_sepconv_blocks(self, biome_dim, only_entry, padding_mode):
        blocks = [
            SepConvBlock(
                in_channels=self.num_sepconv_filters,
                filters=[self.num_sepconv_filters, self.num_sepconv_filters],
                biome_dim=biome_dim,
                only_entry=only_entry,
                padding_mode=padding_mode,
                valid_padding=self.valid_padding
            )
            for _ in range(self.num_sepconv_blocks)
        ]
        return nn.Sequential(*blocks)

    def forward(self, inputs):
        """
        Args:
            x: input tensor: first 12 channels are sentinel-2 bands, last 3 channels are lat lon encoding
        """
        x, b = inputs
        x = self.entry_block(x, b)
        if self.long_skip:
            shortcut = x
        inputs = (x, b)
        x, _ = self.sepconv_blocks(inputs)
        if self.long_skip:
            x = x + shortcut
        predictions = self.predictions(x)

        if self.returns == "dense":
            return predictions if not self.return_latents else (predictions, x)
        elif self.returns == "pixel" :
            return self.pixelwise(predictions)
        else:
            raise ValueError(
                f"XceptionS2 model output is undefined for: returns='{self.returns}'"
            )


# Multi Task Xception ###################################################################################################

class NicoNet_FiLM(pl.LightningModule) :
    """
        Module defining the Multi Task (MT) version of the Xception architecture. It is made of:
    """

    def __init__(self, in_features, num_outputs, emb_dim, biome_dim = 128, num_sepconv_blocks = 8,
                num_sepconv_filters = 728, long_skip = False, returns = "dense", patch_size = 15,
                only_entry = False, linear_emb = False, padding_mode = 'zeros', sigreg_lambda = 0.0):
        """
            - `in_features` (int) : `in_channels` expected by the first layer;
            - `num_outputs` (int) : `out_channels` expected by the last layer of the body;

        """

        super().__init__()
        self.in_features = in_features
        self.intermediary_outputs = num_outputs

        # Linear vs MLP embedding
        if linear_emb :
            self.mlp = nn.Sequential(nn.Linear(emb_dim, biome_dim))
        else:
            self.mlp = nn.Sequential(
                        nn.Linear(emb_dim, biome_dim),
                        nn.GELU(),
                        nn.Linear(biome_dim, biome_dim)
                    )

        # Body
        self.body = XceptionS2_FiLM(in_channels = self.in_features, out_channels = self.intermediary_outputs,
                                    num_sepconv_blocks = num_sepconv_blocks, num_sepconv_filters = num_sepconv_filters,
                                    returns = returns, long_skip = long_skip, manual_init = False, 
                                    freeze_features = False, freeze_last_mean = False, biome_dim = biome_dim,
                                    only_entry = only_entry, valid_padding = True if padding_mode == 'valid' else False,
                                    padding_mode = 'zeros' if padding_mode == 'valid' else 'zeros', patch_size = patch_size,
                                    sigreg_lambda = sigreg_lambda)


    def forward(self, inputs) :
        x, b = inputs
        b = self.mlp(b)
        inputs = (x, b)
        x = self.body(inputs) # (batch_size, intermediary_outputs, size, size)
        return x


if __name__ == '__main__' : 

    #"""
    # Test the UNet FiLM model
    model = NicoNet_FiLM(in_features = 31, num_outputs = 1, emb_dim = 64, num_sepconv_blocks = 6, num_sepconv_filters = 256, returns = 'dense', long_skip = True, only_entry = True, padding_mode = 'valid', patch_size = 15)

    # Test the forward pass
    x = torch.randn(128, 31, 15, 15)
    b = torch.randn(128, 64)
    y = model((x, b))
    print(y.size())

    print('NicoNet FiLM model test passed.')