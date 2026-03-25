import os
import sys

import numpy as np

from .... import math as so3

if __name__ == "__main__":
    np.set_printoptions(linewidth=250, precision=3, suppress=True)

    seq = "".join(["ATCG"[np.random.randint(4)] for i in range(147)])

    num_seqs = 20
    fn = os.path.join(os.path.dirname(__file__), "seqs")

    ##############################################################
    ##############################################################
    ##############################################################
    # gen bps
    seqs = []
    seqs.append(seq)
    print(seq)
    for i in range(num_seqs):
        id = np.random.randint(147)
        prev = seq[id]
        remaining = "ATCG".replace(prev, "")

        nseq = ""
        if id > 0:
            nseq += seq[:id]
        nseq += remaining[np.random.randint(3)]
        if id < len(seq) - 1:
            nseq += seq[id + 1 :]
        seq = nseq
        ph = "".join(" " for i in range(id))
        print(ph + "^")
        print(seq)

        seqs.append(seq)

    with open(fn, "w") as f:
        for seq in seqs:
            f.write(seq + "\n")
