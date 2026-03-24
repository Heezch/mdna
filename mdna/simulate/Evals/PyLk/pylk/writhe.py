import warnings
import numpy as np
from .writhemap import writhemap

def writhe(pos: np.ndarray, closed: bool = True, num_ext: int = 0, ext_dir: np.ndarray = None):
    if num_ext > 0:
        if ext_dir is None:
            raise ValueError(f'writhe: Extension of the chain requires the extesion direction (ext_dir) to be specified.')
        if closed:
            raise ValueError(f'writhe: Chain extension not supported for closed chain.')  
        ext_pos = np.empty((pos.shape[0]+2*num_ext,pos.shape[1]))
        ext_pos[num_ext:-num_ext] = pos
        disc_len = np.linalg.norm(pos[1]-pos[0])
        for i in range(1,num_ext+1):
            ext_pos[num_ext-i] = ext_pos[num_ext] - i*ext_dir*disc_len
            ext_pos[-num_ext-1+i] = ext_pos[-num_ext-1] + i*ext_dir*disc_len
        pos = ext_pos
    wm = writhemap(pos)
    if not closed:
        wm = wm[:-1,:-1]
    return np.sum(wm)