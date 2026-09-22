"""

UPerNet (`--arch upernet`): PANGAEA's regression UPerNet head, wired into this repo's training code.

WHY. The 11-encoder GFM benchmark (see benchmark_pangaea/) trains every foundation model with
PANGAEA's `RegUPerNet` decoder. This module makes that same decoder available as an architecture
HERE, so that AEF / TESSERA / Sentinel-2 inputs can be trained with the benchmark's decoder
instead of NicoNet+FiLM -- i.e. so the two code bases can be compared at equal decoder. The
decoder code itself is the vendored, unmodified copy in `pangaea_decoders/`; everything in this
file is the adapter.

THERE IS NO ENCODER. In PANGAEA the decoder sits on a (frozen or fine-tuned) GFM encoder that
returns a list of feature maps. Here the model is handed an already-dense feature stack
`(B, in_features, H, W)` -- AEF embeddings, TESSERA embeddings, or the stacked
S2/ALOS/DEM/lat-lon/... channels that dataset.py concatenates -- so there is nothing left to
encode. `FeatureStackEncoder` is therefore an IDENTITY encoder (no parameters) that presents
that stack as the `levels` feature levels the FPN expects; all learned capacity is in the
decoder. To put a real GFM encoder under this head, use the PANGAEA fork, not this file.

PYRAMID (`--upernet_pyramid`). PANGAEA's ViT encoders emit tokens at 1/16 of the input
resolution, and `Feature2Pyramid`'s (4, 2, 1, 0.5) rescaling brings them back towards it. Our
features are ALREADY at full resolution, hence two modes:

  * 'flat' (default) -- every level stays at the input resolution (rescales 1, 1, 1, 1). This is
    the honest analogue: there is no lost resolution to recover.
  * 'vit' -- reproduces PANGAEA's (4, 2, 1, 0.5) literally, so the decoder runs at 4x the input
    resolution (a 25x25 patch gives a 100x100 top level, ~ the 56x56 the benchmark's ViTs give).
    Same geometry as the benchmark, but ~27x the activation memory PER SAMPLE. Measured peak
    (RTX 3090, 25x25 patches, in_features=64, channels=512, Adam, fwd+bwd+step):

        flat  0.70 GB @ bs=8   0.85 GB @ bs=16   -> ~0.019 GB/sample  (bs=64 ~ 1.7 GB)
        vit   4.42 GB @ bs=8   8.57 GB @ bs=16   -> ~0.518 GB/sample  (bs=32 ~ 17 GB, bs=64 OOMs 24 GB)

    So 'vit' needs --batch_size lowered to ~32 or below on a 24 GB card.

SUPERVISION. As for every other arch here, the loss reads ONE pixel: the patch centre (see
wrapper.py). The off-centre output of this head is therefore NOT supervised -- any prediction
map built from an `upernet` checkpoint must be centre-pixel only, exactly as for the PANGAEA
runs.

"""

###################################################################################################
# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

from pangaea_decoders.encoder import Encoder
from pangaea_decoders.upernet import RegUPerNet

###################################################################################################
# Identity encoder


class FeatureStackEncoder(Encoder):
    """
    Identity "encoder": presents the input feature stack as the feature levels the FPN expects.

    Args:
     - in_features (int) : number of input channels (== --in_features)
     - patch_size (int) : spatial size of the input patch (informational; the decoder is fully
       convolutional and never relies on it)
     - levels (int) : number of feature levels handed to the decoder (UPerNet uses 4)
     - pyramid (bool) : True -> the decoder leaves every level at the input resolution;
       False -> it applies PANGAEA's (4, 2, 1, 0.5) rescaling (see module docstring)
    """

    def __init__(self, in_features, patch_size=25, levels=4, pyramid=True):
        super().__init__(
            model_name='feature_stack',
            input_bands={},
            input_size=patch_size,
            embed_dim=in_features,
            output_layers=list(range(levels)),
            output_dim=in_features,
            multi_temporal=False,
            multi_temporal_output=False,
            # RegUPerNet reads this to decide the neck's rescales: pyramid_output=True means
            # "the levels already are a pyramid, leave them alone" -> rescales (1, 1, 1, 1).
            pyramid_output=pyramid,
        )

    def forward(self, x):
        # The same tensor at every level: the FPN branches differ only through the neck's
        # rescaling ('vit') or through their own weights ('flat').
        return [x for _ in self.output_layers]


###################################################################################################
# Decoder


class _RegUPerNet(RegUPerNet):
    """
    `RegUPerNet` with two adjustments, both no-ops for the standard single-output regression case:

     - `num_outputs` output channels instead of a hardcoded 1 (needed for GNLL's 2 outputs, and
       for --predict biome's 14). The final ReLU is kept ONLY for num_outputs == 1, where the
       target (agbd / rh98) is non-negative; clamping logits of a 2- or 14-output head would be
       wrong.
     - `forward` takes the input tensor directly instead of PANGAEA's `{modality: tensor}` dict,
       which is what dataset.py hands the model here.

    With num_outputs == 1 the layers and the forward pass are identical to upstream.
    """

    def __init__(self, encoder, finetune, channels, num_outputs=1, pool_scales=(1, 2, 3, 6)):
        super().__init__(encoder=encoder, finetune=finetune, channels=channels, pool_scales=pool_scales)
        self.num_outputs = num_outputs
        self.relu_output = (num_outputs == 1)
        if num_outputs != 1:
            self.conv_reg = nn.Conv2d(self.channels, num_outputs, kernel_size=1)

    def forward(self, img, output_shape=None):
        # Mirrors RegUPerNet.forward(); the encoder has no parameters, so `finetune` only
        # decides whether the identity pass runs under no_grad (gradients reach the decoder
        # either way).
        if not self.finetune:
            with torch.no_grad():
                feat = self.encoder(img)
        else:
            feat = self.encoder(img)

        feat = self.neck(feat)
        feat = self._forward_feature(feat)
        feat = self.dropout(feat)
        output = self.conv_reg(feat)
        if self.relu_output: output = torch.relu(output)

        if output_shape is None: output_shape = img.shape[-2:]
        output = F.interpolate(output, size=output_shape, mode='bilinear')

        return output


###################################################################################################
# Model


class UPerNet(nn.Module):
    """
    PANGAEA's RegUPerNet over a dense input feature stack. See the module docstring.

    Args:
     - in_features (int) : number of input channels
     - num_outputs (int) : number of output channels (1 for agbd / rh98)
     - patch_size (int) : spatial size of the input patch
     - channels (int) : decoder width (PANGAEA's reg_upernet config uses 512)
     - pyramid (str) : 'flat' (levels at the input resolution) or 'vit' (PANGAEA's 4, 2, 1, 0.5)
     - levels (int) : number of feature levels fed to the FPN (UPerNet uses 4)
     - returns (str) : kept for interface parity with the other archs; only 'dense' is supported,
       since the head is fully convolutional and wrapper.py slices the centre pixel itself.
    """

    def __init__(self, in_features, num_outputs=1, patch_size=25, channels=512,
                 pyramid='flat', levels=4, returns='dense'):
        super().__init__()

        if pyramid not in ('flat', 'vit'):
            raise ValueError(f"unknown --upernet_pyramid {pyramid}, expected 'flat' or 'vit'")
        if returns != 'dense':
            raise NotImplementedError(f"upernet only supports returns='dense', got '{returns}'")
        if levels < 1:
            raise ValueError(f'--upernet_levels must be >= 1, got {levels}')

        self.returns = returns
        self.num_outputs = num_outputs
        self.pyramid = pyramid

        self.encoder = FeatureStackEncoder(in_features = in_features, patch_size = patch_size,
                                           levels = levels, pyramid = (pyramid == 'flat'))
        self.model = _RegUPerNet(encoder = self.encoder, finetune = True, channels = channels,
                                 num_outputs = num_outputs)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            # FiLM archs get (images, biome_embs); this head has no conditioning path, and
            # silently dropping the embeddings would train a different model than requested.
            raise NotImplementedError('upernet does not support FiLM conditioning; run it with film=false.')
        return self.model(x, output_shape = x.shape[-2:])
