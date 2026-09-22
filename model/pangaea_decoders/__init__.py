"""

Decoders vendored from PANGAEA (https://github.com/VMarsocci/pangaea-bench), GPL-3.0.

`upernet.py`, `ltae.py` and `base.py` are upstream files, unmodified except that their
`pangaea.{decoders,encoders}.*` imports were rewritten to relative imports so the package
stands alone inside this repo (there is no `pangaea` package here). `encoder.py` is a trimmed
stand-in for upstream's `Encoder` base class -- see its docstring.

The AGBD-side adapter that turns `RegUPerNet` into an `--arch upernet` usable by train.py /
eval.py lives in `model/upernet.py`, not here: this package is kept as close to upstream as
possible so it can be refreshed from the fork.

"""

from .base import Decoder
from .encoder import Encoder
from .upernet import RegUPerNet, SegUPerNet
