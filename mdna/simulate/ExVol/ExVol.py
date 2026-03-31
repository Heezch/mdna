import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..chain import Chain


class ExVol(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def check(self, moved: List = None):
        pass
