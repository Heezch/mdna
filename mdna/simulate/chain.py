import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from .. import math as so3
from .utils import triad_realign

class Chain:
    
    realign_after: int = 100000
    
    def __init__(
        self, configuration: np.ndarray, closed: bool = False, keep_backup: bool = False
    ):
        """
        Initiate:
            - configuration
            - energy model (separate object passed to chain)

        """

        self.conf = configuration
        self.closed = closed
        self.keep_backup = keep_backup
        self.set_backup(self.conf)
        
        self.temp = 300

        self.nbp = len(self.conf)
        self.nbps = self.nbp
        if not self.closed:
            self.nbps -= 1
        
        self.realign_step_count = 0
        self.realign_triads()
        
            
    def set_conf(self, conf: np.ndarray) -> None:
        self.conf = conf
        self.realign_triads()
        self.set_backup()

    def activate_backup(self) -> None:
        self.keep_backup = True
        self.set_backup(self.conf)
    
    def set_backup(self, conf: np.ndarray = None) -> None:
        if self.keep_backup:
            if conf is None:
                self.backup_conf = np.copy(self.conf)
            else:
                self.backup_conf = np.copy(conf)

    def revert_to_backup(self) -> None:
        self.conf = np.copy(self.backup_conf)

    def realign_triads(self) -> None:
        for i in range(self.nbp):
            # self.conf[i,:3,:3] = so3.euler2rotmat(so3.rotmat2euler(self.conf[i,:3,:3]))
            self.conf[i,:3,:3] = triad_realign(self.conf[i,:3,:3])
        self.set_backup()
    
    def check_realign(self) -> bool:
        self.realign_step_count += 1
        if self.realign_step_count < self.realign_after:
            return False
        self.realign_step_count = 0
        self.realign_triads()
        return True
    
    @property
    def positions(self):
        return self.conf[:,:3,3]
    
    @property
    def triads(self):
        return self.conf[:,:3,:3]

##################################################################################################
### Extra Chain Methods ##########################################################################
##################################################################################################

def se3_triads(triads: np.ndarray, positions: np.ndarray):
    if len(positions) != len(triads):
        raise ValueError(
            f"Mismatched dimensions of positions ({positions.shape}) and triads ({triads.shape})."
        )
    conf = np.zeros((len(positions), 4, 4))
    conf[:, :3, :3] = triads
    conf[:, :3, 3] = positions
    conf[:, 3, 3] = 1
    return conf