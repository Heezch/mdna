import numpy as np
import mdtraj as md
import matplotlib as mpl
import matplotlib.font_manager as font_manager

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext'] = True

def _sequence_letters(traj):
    """List with nucleobase letter codes of first strand"""
    bases = ['DA','DT','DG','DC']  
    sequence = [b[1] for res in traj.topology._residues for b in bases if b in str(res)]
    return sequence[:int(len(sequence)/2)]

def _base_pair_letters(traj):
    """Letter code of base pairs as list, e.g., ['A-T', ...]"""
    bases = ['DA','DT','DG','DC']  
    sequence = [b[1] for res in traj.topology._residues for b in bases if b in str(res)]
    a = sequence[:int(len(sequence)/2)]
    b = sequence[int(len(sequence)/2):]
    return [f'{i}-{j}'for i,j in zip(a,reversed(b))]

def _compute_distance(xyz, pairs):
    "Distance between pairs of points in each frame"
    delta = np.diff(xyz[:, pairs], axis=2)[:, :, 0]
    return (delta ** 2.).sum(-1) ** 0.5