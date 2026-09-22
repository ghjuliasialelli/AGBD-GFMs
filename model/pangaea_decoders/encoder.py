"""

Minimal stand-in for `pangaea.encoders.base.Encoder`.

The decoders vendored in this package (`upernet.py`) are taken verbatim from PANGAEA, so they
type-annotate and read attributes off an `Encoder`. Upstream's base class also carries the
weight-download machinery (gdown / urllib / tqdm), which pulls in dependencies this repo does
not have and does nothing useful here -- nothing in `upernet.py` calls it. This is therefore a
trimmed copy holding ONLY what the decoders actually read:

    output_dim, output_layers, pyramid_output, multi_temporal, multi_temporal_output,
    enforce_single_temporal()

The AGBD adapter (`model/upernet.py`) subclasses this; a real GFM encoder would come from the
PANGAEA fork instead (see benchmark_pangaea/README.md).

"""

import torch.nn as nn


class Encoder(nn.Module):
    """Base class for encoders. Trimmed copy of `pangaea.encoders.base.Encoder` (see module docstring)."""

    def __init__(
        self,
        model_name: str,
        input_bands: dict,
        input_size: int,
        embed_dim: int,
        output_layers: list,
        output_dim,
        multi_temporal: bool = False,
        multi_temporal_output: bool = False,
        pyramid_output: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.input_bands = input_bands
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.output_layers = output_layers
        # Upstream semantics: an int is broadcast to one entry per output layer.
        self.output_dim = (
            [output_dim for _ in output_layers] if isinstance(output_dim, int) else list(output_dim)
        )
        self.multi_temporal = multi_temporal
        self.multi_temporal_output = multi_temporal_output
        self.pyramid_output = pyramid_output

    def enforce_single_temporal(self):
        return

    def forward(self, x):
        raise NotImplementedError
