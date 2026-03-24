from __future__ import annotations
import numpy as np
import scipy as sp
import sys
from typing import List, Tuple, Callable, Any, Dict

# DOES NOT SUPPORT NEGATIVE INDEXING TO AS COUNTED BACKWARDS FROM THE END
# TODO: For periodic boundary condition implement averaging in all directions.
#       Currently averaging is done only along the diagonal


class BlockOverlapMatrix:
    # ndims: int
    # average: bool
    # matblocks: List[BOMat]
    # ranges: List[List[int]]

    # xlo: int
    # xhi: int
    # ylo: int
    # yhi: int

    # periodic: bool
    # fixed_size: bool
    # check_bounds: bool
    # check_bounds_on_read: bool

    # xrge: int
    # yrge: int
    # shape: tuple[int, int]

    ###################################################################################

    def __init__(
        self,
        ndims: int,
        average: bool = True,
        xlo: int = None,
        xhi: int = None,
        ylo: int = None,
        yhi: int = None,
        periodic: bool=False,
        fixed_size: bool=False,
        check_bounds: bool=True,
        check_bounds_on_read: bool=True
    ):
        self.ndims = ndims
        self.average = average
        self.matblocks = list()

        if None in [xlo, xhi, ylo, yhi]:
            if fixed_size:
                raise ValueError(
                    "For fixed size matrix all bounds need to be specified!"
                )
            if periodic:
                raise ValueError("For periodic matrix all bounds need to be specified!")

        self.fixed_size = fixed_size
        self.periodic = periodic
        if periodic:
            self.fixed_size = False
        self.check_bounds = check_bounds
        self.check_bounds_on_read = check_bounds_on_read

        def set_val(x):
            if x is None:
                return 0
            else:
                return x

        self.xlo = set_val(xlo)
        self.xhi = set_val(xhi)
        self.ylo = set_val(ylo)
        self.yhi = set_val(yhi)

        self.xrge = self.xhi - self.xlo
        self.yrge = self.yhi - self.ylo
        self.shape = (self.xrge, self.yrge)

    ###################################################################################

    def _convert_bounds(
        self, x1: int, x2: int, y1: int | None, y2: int | None
    ) -> Tuple[int, int, int, int]:
        if y1 is None:
            y1 = x1
        if y2 is None:
            y2 = x2
        if self.periodic:
            shift_x1 = (x1 - self.xlo) % self.xrge + self.xlo
            dx1 = shift_x1 - x1
            x1 = shift_x1
            x2 = x2 + dx1

            shift_y1 = (y1 - self.ylo) % self.yrge + self.ylo
            dy1 = shift_y1 - y1
            y1 = shift_y1
            y2 = y2 + dy1
        return x1, x2, y1, y2

    ###################################################################################

    def _valid_arg_order(self, x1: int, x2: int, y1: int, y2: int) -> bool:
        if x1 >= x2:
            raise ValueError(
                f"lower bound x1 ({x1}) needs to be strictly smaller than upper bound x2 ({x2}). Negative indexing relative to \
                             upper array bound is not supported!"
            )
        if y1 >= y2:
            raise ValueError(
                f"lower bound y1 ({y1}) needs to be strictly smaller than upper bound y2 ({y2}). Negative indexing relative to \
                             upper array bound is not supported!"
            )

    def _check_bounds(self, x1: int, x2: int, y1: int, y2: int) -> None:
        if self.fixed_size and self.check_bounds:
            if x1 < self.xlo:
                raise ValueError(f"x1 ({x1}) is out of bounds with xlo={self.xlo}.")
            if x2 > self.xhi:
                raise ValueError(f"x2 ({x2}) is out of bounds with xhi={self.xhi}.")
            if y1 < self.ylo:
                raise ValueError(f"y1 ({y1}) is out of bounds with ylo={self.ylo}.")
            if y2 > self.yhi:
                raise ValueError(f"y2 ({y2}) is out of bounds with yhi={self.yhi}.")

    ###################################################################################

    def _update_bounds(self, x1: int, x2: int, y1: int, y2: int) -> None:
        if self.periodic or self.fixed_size:
            return
        # update bounds
        if x1 < self.xlo:
            self.xlo = x1
        if x2 > self.xhi:
            self.xhi = x2
        if y1 < self.ylo:
            self.ylo = y1
        if y2 > self.yhi:
            self.yhi = y2
        self.xrge = self.xhi - self.xlo
        self.yrge = self.yhi - self.ylo
        self.shape = (self.xrge, self.yrge)

    ###################################################################################

    def _slice2ids(
        self, ids: Tuple[slice]
    ) -> Tuple[int, int, int, int]:
        x1 = ids[0].start
        x2 = ids[0].stop
        y1 = ids[1].start
        y2 = ids[1].stop

        # print(x1,x2)
        # print(y1,y2)

        if x1 == None:
            x1 = self.xlo
        if x2 == None:
            x2 = self.xhi
        if y1 == None:
            y1 = self.ylo
        if y2 == None:
            y2 = self.yhi

        x1, x2, y1, y2 = self._convert_bounds(x1, x2, y1, y2)
        self._valid_arg_order(x1,x2,y1,y2)
        # self._check_bounds(x1, x2, y1, y2)
        return x1, x2, y1, y2

    ###################################################################################

    def __len__(self) -> Tuple[int, int]:
        return self.xhi - self.xlo

    def __contains__(self, elem: BOMat) -> bool:
        return elem in self.matblocks

    ###################################################################################

    def _new_block(
        self,
        assigntype: str,
        mat: np.ndarray,
        x1: int,
        x2: int,
        y1: int = None,
        y2: int = None,
        image=False,
    ) -> BOMat:
        assert assigntype in [
            "add",
            "set",
        ], f'unknown assigntype "{assigntype}". Needs to be either "set" or "add".'

        new_block = BOMat(
            mat,
            x1,
            x2,
            y1,
            y2,
            image=image,
            periodic=self.periodic,
            xrge=self.xrge,
            yrge=self.yrge,
        )
        self.matblocks.append(new_block)
        return new_block

    ###################################################################################

    def __setitem__(
        self, ids: Tuple[slice] | int, mat: np.ndarray | float | int
    ) -> None:
        if not isinstance(ids, tuple):
            raise ValueError(
                f"Expected tuple of two slices, but received argument of type {type(ids)}."
            )
        for sl in ids:
            if not isinstance(sl, slice):
                raise ValueError(f"Expected slice but encountered {type(sl)}.")

        x1, x2, y1, y2 = self._slice2ids(ids)
        self._check_bounds(x1, x2, y1, y2)

        # if mat is scalar generate unform matrix of that scalar
        if not isinstance(mat, np.ndarray):
            try:
                val = float(mat)
            except:
                raise ValueError("mat should be a scalar or numpy ndarray")
            mat = np.ones((x2 - x1, y2 - y1)) * val

        new_block = self._new_block("set", mat, x1, x2, y1=y1, y2=y2, image=False)
        self._update_bounds(x1, x2, y1, y2)

        # set values in existing blocks
        for block in self.matblocks:
            if block == new_block:
                continue

            extr_mat = np.zeros((block.x2 - block.x1, block.y2 - block.y1))
            extr_cnt = np.zeros(extr_mat.shape)
            extr_mat, extr_cnt = new_block.extract(
                extr_mat,
                extr_cnt,
                block.x1,
                block.x2,
                block.y1,
                block.y2,
                use_weight=False,
            )
            block.mat = block.mat * (1 - extr_cnt) + extr_mat

    ###################################################################################

    def add_block(
        self, mat: np.ndarray, x1: int, x2: int, y1: int = None, y2: int = None
    ) -> bool:
        x1, x2, y1, y2 = self._convert_bounds(x1, x2, y1, y2)
        self._valid_arg_order(x1,x2,y1,y2)
        self._check_bounds(x1, x2, y1, y2)
        self._new_block("add", mat, x1, x2, y1=y1, y2=y2, image=False)
        self._update_bounds(x1, x2, y1, y2)

    ###################################################################################

    def __getitem__(self, ids: Tuple[slice] | int) -> float | np.ndarray:
        x1, x2, y1, y2 = self._slice2ids(ids)
        if self.check_bounds_on_read:
            self._check_bounds(x1, x2, y1, y2)
        
        mat = np.zeros((x2 - x1, y2 - y1))
        cnt = np.zeros(mat.shape)
        for block in self.matblocks:
            mat, cnt = block.extract(mat, cnt, x1, x2, y1, y2)
        cnt[cnt == 0] = 1
        return mat / cnt

    def to_array(self):
        return self[self.xlo : self.xhi, self.ylo : self.yhi]
    
    def __mul__(self, B):
        if hasattr(B, "__len__"):
            raise ValueError(f'Matrix multiplication is currently not supported for instances of BlockOverlapMatrix.')
        for block in self.matblocks:
            block.mat *= B
        return self
    
    def __rmul__(self, B):
        if hasattr(B, "__len__"):
            raise ValueError(f'Matrix multiplication is currently not supported for instances of BlockOverlapMatrix.')
        for block in self.matblocks:
            block.mat *= B
        return self
    

#######################################################################################
#######################################################################################
#######################################################################################


class BOMat:
    # mat: np.ndarray
    # x1: int
    # x2: int
    # y1: int
    # y2: int
    # overlap_mat: np.ndarray
    # image: bool

    # periodic: bool
    # xrge: int
    # yrge: int

    # # TO DO: Make this a weight profile, i.e. a matrix with individual weights. This will allow to combine blocks of different sizes to
    # # optimize memory use.
    # weight: int

    def __init__(
        self,
        mat: np.ndarray,
        x1: int,
        x2: int,
        y1: int | None = None,
        y2: int | None = None,
        copy=True,
        image=False,
        periodic: bool = False,
        xrge: int = 0,
        yrge: int = 0,
        weight: int = 1,
    ):
        self.image = image
        if copy:
            self.mat = np.copy(mat)
        else:
            self.mat = mat
        self.x1 = x1
        self.x2 = x2
        if y1 is None:
            self.y1 = x1
        else:
            self.y1 = y1
        if y2 is None:
            self.y2 = x2
        else:
            self.y2 = y2
        if len(mat) != self.x2 - self.x1:
            raise IndexError(f"Size of x-dimension ({len(mat)}) inconsistent with specified x range ({self.x2 - self.x1})")
        if len(mat[0]) != self.y2 - self.y1:
            raise IndexError(f"Size of x-dimension ({len(mat[0])}) inconsistent with specified x range ({self.y2 - self.y1})")
        self.overlap_mat = np.ones(self.mat.shape)

        self.periodic = periodic
        self.xrge = xrge
        self.yrge = yrge

        self.weight = weight

    def extract(
        self,
        extr_mat: np.ndarray,
        cnt: np.ndarray,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        use_weight=True,
    ) -> Tuple[np.ndarray, int, int, int, int]:
        xshifts = self._periodic_shifts(self.x1, self.x2, x1, x2, self.xrge)
        yshifts = self._periodic_shifts(self.y1, self.y2, y1, y2, self.yrge)

        if not self.periodic:
            if 0 not in xshifts or 0 not in yshifts:
                return extr_mat, cnt
            return self._extract(extr_mat, cnt, x1, x2, y1, y2, use_weight=use_weight)

        for xshift in xshifts:
            for yshift in yshifts:
                extr_mat, cnt = self._extract(
                    extr_mat,
                    cnt,
                    x1 - xshift,
                    x2 - xshift,
                    y1 - yshift,
                    y2 - yshift,
                    use_weight=use_weight,
                )
        return extr_mat, cnt

    def _extract(
        self,
        extr_mat: np.ndarray,
        cnt: np.ndarray,
        x1: int,
        x2: int,
        y1: int,
        y2: int,
        use_weight=True,
    ):
        # assert (x1 <= self.x1 <= x2) or (x1 <= self.x2 <= x2), "x out of range"
        # assert (y1 <= self.y1 <= y2) or (y1 <= self.y2 <= y2), "y out of range"

        if self.x1 <= x1:
            xlo = x1
        else:
            xlo = self.x1
        if self.x2 >= x2:
            xhi = x2
        else:
            xhi = self.x2

        if self.y1 <= y1:
            ylo = y1
        else:
            ylo = self.y1
        if self.y2 >= y2:
            yhi = y2
        else:
            yhi = self.y2
        if use_weight:
            weight = self.weight
        else:
            weight = 1

        extr_mat[xlo - x1 : xhi - x1, ylo - y1 : yhi - y1] += (
            self.mat[xlo - self.x1 : xhi - self.x1, ylo - self.y1 : yhi - self.y1]
            * weight
        )
        cnt[xlo - x1 : xhi - x1, ylo - y1 : yhi - y1] += weight
        return extr_mat, cnt

    def _periodic_shifts(self, x1: int, x2, b1: int, b2: int, rge: int) -> List[int]:
        sx2 = (x2 - b1) % rge + b1
        baseshift = sx2 - x2
        sx1 = x1 + baseshift
        num = (b2 - 1 - sx1) // rge
        shifts = [baseshift + (i + 1) * rge for i in range(num)]
        if sx1 < b2:
            shifts += [baseshift]
        return shifts


if __name__ == "__main__":
    m1 = np.ones((4, 4))
    m2 = np.ones((4, 4)) * 0
    m3 = np.ones((4, 4)) * -1

    np.set_printoptions(linewidth=300, precision=4)

    block1 = BOMat(m1, 0, 4)
    block2 = BOMat(m2, 2, 6)

    bom = BlockOverlapMatrix(
        3, average=True, periodic=True, xlo=0, xhi=32, ylo=0, yhi=32, fixed_size=True
    )

    bom.add_block(m1, 0, 4)
    # print(bom[-2:10, -2:10])
    # print(bom.to_array())

    bom.add_block(m2, 2, 6)
    # print(bom[-2:10, -2:10])
    # print(bom.to_array())

    bom.add_block(m3, 5, 9)

    # print(bom.to_array())
    # print(bom[:16,:16])
    # sys.exit()

    for i in range(1, 2):
        bom.add_block(m1, 0 + i * 8, 4 + i * 8)
        bom.add_block(m2, 2 + i * 8, 6 + i * 8)
        bom.add_block(m3, 5 + i * 8, 9 + i * 8)

    bom[3:12, 3:12] = 6
    # bom.add_block(m2,8,12)

    print(f"num_blocks = {len(bom.matblocks)}")

    # print(bom[-2:10, -2:10])
    print("to array")
    print(bom.to_array())
    print("400")
    print(bom[:16, :16])



    # bom.add_block(m1, -2, 2)
    # print(bom[0:12, 0:12])
    # print(bom.to_array())

    # bom[5:10, 5:10] = np.ones((5,) * 2) * 2
    # print(bom[-2:10, -2:10])
    # print(bom.to_array())

    # bom.add_block(m3,-4,0)
    # print(bom.to_array())
