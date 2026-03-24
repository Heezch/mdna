import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..BPStep.BPS_RBP import RBP
from ..chain import Chain
# from ..BPStep.BPStep import BPStep

def init_bps(
    model: str, chain: Chain, sequence: str, closed: bool = False, temp: float = 300
):
    if model.lower() in ["lankas", "rbp_md"]:
        specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}
        return RBP(chain, sequence, specs, closed=closed, static_group=True, temp=temp)