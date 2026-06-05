from nico_net_film import NicoNet_FiLM
from mlp import MLP
from lp import LP

from biomes import REF_BIOMES
import torch.nn as nn

###################################################################################################
# Wrapper #########################################################################################
###################################################################################################

class Net(nn.Module):
    """
    This class is a wrapper around the different models.

    NOTE: trimmed for the AGB-GFM release. Only the architectures used in the paper
    are kept: linear probing ('lp'), MLP ('mlp'), and the supervised SOTA NicoNet+FiLM
    ('nico_film'). The dropped archs (nico, nico_gaussian, fcn, unet, unet_gaussian,
    unet_film, effunet) and their modules were removed.
    """
    def __init__(self, model_name, emb_dim, in_features = 4, num_outputs = 1, downsample = None,
                 patch_size = [15, 15], pretrained_path = None, local = False, device = 'cpu',
                 biome_dim = 128, num_sepconv_blocks = 8, num_sepconv_filters = 728, long_skip = False,
                 only_entry = False, linear_emb = False, padding_mode = 'zeros', returns = "dense",
                 sigreg_lambda = 0.0, predict = 'agbd'):
        super(Net, self).__init__()

        self.model_name = model_name
        self.num_outputs = num_outputs
        self.pretrained_path = pretrained_path
        self.biomes = list(REF_BIOMES.keys())
        self.returns = returns
        self.predict = predict
        self.pool = nn.Identity()

        # Linear probing
        if self.model_name == 'lp' :
            self.model = LP(input_dim = in_features, output_dim = num_outputs)

        # MLP
        elif self.model_name == 'mlp' :
            self.model = MLP(input_dim = in_features, hidden_dim = 256, output_dim = num_outputs)

        # Nico net, with FiLM (supervised SOTA)
        elif self.model_name == "nico_film":
            self.model = NicoNet_FiLM(in_features = in_features, num_outputs = num_outputs, emb_dim = emb_dim, biome_dim = biome_dim,
                                        num_sepconv_blocks = num_sepconv_blocks, num_sepconv_filters = num_sepconv_filters,
                                        long_skip = long_skip, returns = returns, patch_size = patch_size[0], only_entry = only_entry,
                                        linear_emb = linear_emb, padding_mode = padding_mode, sigreg_lambda = sigreg_lambda)

        else:
            raise NotImplementedError(f'unknown model name {model_name}')

    def forward(self, x):
        y = self.model(x)
        if isinstance(y, tuple): y, latents = y
        y = self.pool(y)
        return y if not isinstance(y, tuple) else (y, latents)
