import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ...aux import params2conf
from ...BPStep.BPS_RBP import RBP
from ...BPStep.BPStep import BPStep
from ...chain import Chain
from ...Dumps.xyz import write_xyz
from ...ExVol.ExVol import ExVol
from ...MCStep.clustertranslation import ClusterTrans
from ...MCStep.crankshaft import Crankshaft
from ...MCStep.midstepmove import MidstepMove
from ...MCStep.pivot import Pivot
from ...MCStep.singletriad import SingleTriad
from .... import math as so3
from .read_nuc_conf import read_nucleosome_triads


def calculate_midstep_triads(
    triad_ids: List[
        int
    ],  # index of the lower (left-hand) triad neighboring the constraint midstep-triad
    nucleosome_triads: np.ndarray,
) -> np.ndarray:
    midstep_triads = np.zeros((len(triad_ids), 4, 4))
    for i, id in enumerate(triad_ids):
        T1 = nucleosome_triads[id]
        T2 = nucleosome_triads[id + 1]
        midstep_triads[i, :3, :3] = T1[:3, :3] @ so3.euler2rotmat(
            0.5 * so3.rotmat2euler(T1[:3, :3].T @ T2[:3, :3])
        )
        midstep_triads[i, :3, 3] = 0.5 * (T1[:3, 3] + T2[:3, 3])
        midstep_triads[i, 3, 3] = 1
    return midstep_triads


if __name__ == "__main__":
    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    triadfn = os.path.join(os.path.dirname(__file__), "State/Nucleosome.state")
    conf = read_nucleosome_triads(triadfn)

    midstep_ids = [
        2,
        6,
        14,
        17,
        24,
        29,
        34,
        38,
        45,
        49,
        55,
        59,
        65,
        69,
        76,
        80,
        86,
        90,
        96,
        100,
        107,
        111,
        116,
        121,
        128,
        131,
        139,
        143,
    ]

    misteps = calculate_midstep_triads(midstep_ids, conf)

    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(147)])
    seq601 = "ATCGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACGTACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGATATATACATCCGAT"
    seq = seq601

    closed = False
    specs = {"method": "MD", "gs_key": "group_gs", "stiff_key": "group_stiff"}

    chain = Chain(conf, keep_backup=True, closed=closed)
    bps = RBP(chain, seq, specs, closed=closed, static_group=True)

    moves = []
    # add midstep moves
    msm = MidstepMove(chain, bps, midstep_ids)
    for i in range(14):
        moves.append(msm)

    # add cluster translation moves and crankshaft moves
    for i in range(len(midstep_ids) - 1):
        if midstep_ids[i + 1] - midstep_ids[i] >= 5:
            idfrom = midstep_ids[i] + 2
            idto = midstep_ids[i + 1]
            print(idfrom, idto)
            moves.append(
                ClusterTrans(chain, bps, 2, 8, range_id1=idfrom, range_id2=idto)
            )
            moves.append(Crankshaft(chain, bps, 2, 8, range_id1=idfrom, range_id2=idto))

    # moves = [Crankshaft(chain,bps,2,8,range_id1=19,range_id2=24)]

    # add single moves
    ids = [0, 1]
    for i in range(len(midstep_ids) - 1):
        ids += [i for i in range(midstep_ids[i] + 2, midstep_ids[i + 1])]
    ids += [145, 146]
    stm = SingleTriad(chain, bps, selected_triad_ids=ids)
    for i in range(14):
        moves.append(stm)

    ##############################################################
    ##############################################################
    ##############################################################

    Es = []
    confs = []
    confs.append(np.copy(chain.conf[:, :3, 3]))

    import time

    t1 = time.time()

    for i in range(10000):
        for move in moves:
            move.mc()

        if i % 100 == 0:
            confs.append(np.copy(chain.conf[:, :3, 3]))

        if i % 1000 == 0:
            print(f"step {i}: ")
            for move in set(moves):
                print(f"{move.name}: {move.acceptance_rate()}")
            Es.append(bps.get_total_energy())
            # confs.append(np.copy(chain.conf[:,:3,3]))
            if not bps.check_deform_consistency():
                print("inconsistent")
                sys.exit()

            test_midsteps = calculate_midstep_triads(midstep_ids, chain.conf)

            if np.abs(np.sum(misteps - test_midsteps)) > 1e-10:
                raise ValueError(f"Midsteps not conserved")

    t2 = time.time()
    print(f"dt = {t2-t1}")

    print(f"<E> / DoFs = {np.mean(Es)/(len(conf)-1)/6}")
    types = ["C" for i in range(chain.nbp)]
    # types[-2] = 'O'
    data = {"pos": confs, "types": types}
    write_xyz("nuc.xyz", data)
