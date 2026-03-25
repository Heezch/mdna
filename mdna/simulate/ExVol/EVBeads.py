import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from ..._jit import cond_jit
from .ExVol import ExVol
from ..chain import Chain

from .EVBeads_additional import doubleMove, singleMove 
from .EVBeads_additional import doubleMove_intervals, singleMove_intervals 
from .EVBeads_additional import largest_dist 
from .EVBeads_additional import debug_plot


class EVBeads(ExVol):
    
    debug: bool = False
    quick_overlap_check: bool = True
    numba_interval_checks: bool = True
    
    
    def __init__(
        self, 
        chain: Chain,
        ev_distance: float,
        max_distance: float,
        check_crossings: bool = True,
        ):

        self.chain = chain
        if not self.chain.keep_backup:
            self.chain.activate_backup()
        self.conf = chain.conf
        self.num_bp  = len(self.conf)
        self.closed  = chain.closed
        self.EV_dist = ev_distance
        self.maxdist = max_distance
        self.counter = 0
        self.counter_reject = 0
        self.check_crossings = check_crossings
                
        #################################################################################
        ## Check Validity of 
        if ev_distance <= 0:
            raise ValueError(f'EVBeads: ev_distance should be larger than 0')
        if max_distance <= 0:
            raise ValueError(f'EVBeads: max_distance should be larger than 0')
        if self.EV_dist < self.maxdist:
            raise ValueError(f'EVBeads: ev_distance should be larger than max_distance')

        #################################################################################
        ## Init EV_beads
        self.num_bp_per_EV    = int(self.EV_dist/self.maxdist)
        self.upper_shift      = self.num_bp_per_EV-1
        self.eff_size_EV_bead = self.num_bp_per_EV*self.maxdist
        self.num_EV           = int(np.ceil(self.num_bp/self.num_bp_per_EV))
        self.EV_beads         = np.arange(0,self.num_EV)*self.num_bp_per_EV
        
        self.curr_size_EV_bead = self.eff_size_EV_bead
        self.max_EV_bead_dist  = self.num_bp_per_EV * self.maxdist
        
        #################################################################################
        # If there is sufficient overlap between the excluded regions of neighboring EV_beads,
        # i.e. if next nearest neighbors overlap for a 90deg angle between the two intermediate
        # tangents, excluded volume checks are skipped for the first two neighbours rather than
        # just the nearest neighbours.

        self.neighbour_skip = 1
        if (2*self.maxdist**2*self.num_bp_per_EV**2 < self.EV_dist**2):
            self.neighbour_skip = 2
        self.neighbour_skip_plus_one          = self.neighbour_skip+1
        self.neighbour_skip_boundary          = self.neighbour_skip
        self.neighbour_skip_boundary_plus_one = self.neighbour_skip_boundary+1

        #################################################################################
        # If the total number of bp is not a multiple of the number of bp per EV_bead and the
        # chain has a closed topology, i.e. the first bead is connected to the last, overlap
        # between beads displaced by less than (neighbour_skip+1), which can occure around the
        # periodic boundary, will be discarded. This is done to prevent false overlap detections.
        # Instead, additional static pair checks will be conducted to prevent potential crossings
        # in this region.

        self.additional_boundcheck=False
        if (self.num_bp%self.num_bp_per_EV!=0 and chain.closed):
            print('EV_bead mismatch: including additional boundary checks.')
            self.additional_boundcheck=True
            self.addboundpairs = []
            for i in range(self.neighbour_skip+1):
                self.addboundpairs.append([i*self.num_bp_per_EV,self.num_bp-(self.neighbour_skip+1-i)*self.num_bp_per_EV])
            self.neighbour_skip_boundary += 1
            self.neighbour_skip_boundary_plus_one += 1
            self.addboundpairs = np.array(self.addboundpairs)
        else:
            self.addboundpairs = np.zeros((2,2))
        
        # #################################################################################
        # # Init nlargest 
        # self.num_largest = 5 
        # self.init_nlargest()
        
        print('')
        print("######################################")
        print("#### INITIALIZING EXCLUDED VOLUME ####")
        print("######################################")
        print(f" Excluded Volume Beads: ")
        print(f"   number of EV beads: {self.num_EV}")
        print(f"   bp per EV bead:     {self.num_bp_per_EV}")
        print(f"   Effective size:     {np.round(self.eff_size_EV_bead,decimals=3)}")
        print(f"   Exclusion distance: {self.EV_dist}")
        print("######################################")

        self.EV_FROM = 0
        self.EV_TO   = 1
        self.EV_TYPE = 2
        
        self.EV_TYPE_A = 0
        self.EV_TYPE_B = 1
        self.EV_TYPE_C_FROM = 1000
        self.EV_TYPE_C_TO   = 1999
        self.EV_TYPE_D_FROM = 2000
        self.EV_TYPE_D_TO   = 2999
        self.EV_TYPE_E = 2
        
        # check for overlap in initial configuration
        self.bp_pos = chain.conf[:,:3,3]
        
        if self.EV_dist > 0 and self.check_overlap():
            raise ValueError(f'EVBeads: Overlap detected in intial configuration!')


    #########################################################################################
    ########## Additional Methods ###########################################################
    #########################################################################################
    
    def bp2EV_upper(self,bp_id: int) -> int:
        """
        returns the id of the first EV bead that has a bp index higher or
        equal bp_id
        """
        return (bp_id+self.upper_shift)//self.num_bp_per_EV

    def bp2EV_lower(self,bp_id: int) -> int:
        """
        returns the id of the first EV bead that has a bp index smaller or
        equal bp_id
        """
        return bp_id//self.num_bp_per_EV

    def interval_bp2EV(self, bp_interval: np.ndarray) -> List:
        """self.evpos
        Transforms a bp interval into an EV interval. This interval contains all the EV
        beads for which the reference bp (by which the positions of the EV is defined)
        lies inside the given bp interval.
        """
        return [self.bp2EV_upper(bp_interval[0]),self.bp2EV_lower(bp_interval[1]),bp_interval[2]]
    
    #########################################################################################
    ########## Type Recognition and Assignment ##############################################
    #########################################################################################
    
    def within_EV_typeA(self,id: int) -> bool:
        return id == self.EV_TYPE_A
    def within_EV_typeB(self,id: int) -> bool:
        return id == self.EV_TYPE_B
    def within_EV_typeC(self,id: int) -> bool:
        return self.EV_TYPE_C_FROM <= id and id <= self.EV_TYPE_C_TO
    def within_EV_typeD(self,id: int) -> bool:
        return self.EV_TYPE_D_FROM <= id and id <= self.EV_TYPE_D_TO
    def within_EV_typeE(self,id: int) -> bool:
        return id == self.EV_TYPE_E
    
    def cal_EV_intervals(self, moved: np.ndarray) -> Tuple[List,List,List,List,List]:
        """  Converts the bp intervals into EV intervals and corrects for potential overlap
        - It is critical that the intervals in 'moved' are ordered from left to right.
        TODO: remove the requirement for the moved intervals to be ordered.
        """
        EV_typeA = []
        EV_typeB = []
        EV_typeC = []
        EV_typeD = []
        EV_typeE = []

        # conversion
        EV_intervals = []
        for i in range(len(moved)):
            if moved[i][self.EV_TYPE] >= 0 and moved[i][self.EV_FROM] <= moved[i][self.EV_TO]:
                new_EV_interval = self.interval_bp2EV(moved[i])
                if (new_EV_interval[0] <= new_EV_interval[1] ):
                    EV_intervals.append(new_EV_interval)
        
        # sort intervals
        for i in range(len(EV_intervals)):
            if (EV_intervals[i][self.EV_FROM] <= EV_intervals[i][self.EV_TO]):
                if ( self.within_EV_typeA( EV_intervals[i][self.EV_TYPE] )):
                    EV_typeA.append(EV_intervals[i])
                    continue
                if ( self.within_EV_typeB( EV_intervals[i][self.EV_TYPE] )):
                    EV_typeB.append(EV_intervals[i])
                    continue
                if ( self.within_EV_typeC( EV_intervals[i][self.EV_TYPE] )):
                    EV_typeC.append(EV_intervals[i])
                    continue
                if ( self.within_EV_typeD( EV_intervals[i][self.EV_TYPE] )):
                    EV_typeD.append(EV_intervals[i])
                    continue
                if ( self.within_EV_typeE( EV_intervals[i][self.EV_TYPE] )):
                    EV_typeE.append(EV_intervals[i])
                    continue
        # ##########################################
        # # DEBUG: test interval consistency #######
        # if self.debug:
        #     for i in range(1,len(EV_intervals)):
        #         if (EV_intervals[i][self.EV_FROM] != EV_intervals[i-1][self.EV_TO]+1):
        #             print(f'intervals inconsistent')
        #             for interval in EV_intervals:
        #                 print(interval)
        #             for mov in moved:
        #                 print(mov)
        #             break
        #     if (EV_intervals[-1][1] != self.num_EV-1):
        #         print(f'intervals inconsistent! (last)')
        #         print(f'Index of last EV bead: {self.num_EV-1}')
        #         for interval in EV_intervals:
        #             print(interval)
        #         for mov in moved:
        #             print(mov)
        # ##########################################
        return EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE
        
    #########################################################################################
    ########## Main Check ###################################################################
    #########################################################################################
    
    def check(self, moved: List = None):
                
        self.counter += 1
        EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE = self.cal_EV_intervals(moved)

        # print('#####################')
        # print(f'{moved=}')
        # print(f'{EV_typeA=}')
        # print(f'{EV_typeB=}')
        # print(f'{EV_typeC=}')
        # print(f'{EV_typeD=}')
        # print(f'{EV_typeE=}')
        
        # assign positions
        self.bp_pos        = self.chain.conf[:,:3,3]
        self.bp_pos_backup = self.chain.backup_conf[:,:3,3]
        self.evpos = self.get_evpos(self.chain.conf[:,:3,3])
        # self.evpos_backup = self.get_evpos(self.chain.backup_conf[:,:3,3])
        
        # set current largest distance
        largest = largest_dist(self.evpos,self.closed)
        if largest > self.max_EV_bead_dist:
            # print('Exceeded max dist')
            # print(f'largest dist:     {largest}')
            # print(f'max_EV_bead_dist: {self.max_EV_bead_dist}')
            return False
        
        # check crossings
        if (self.check_crossings):
            check = self.check_intervals(EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE)
        else:
            raise ValueError(f'check_intervals_simpleoverlap not yet implemented')
            check = self.check_intervals_simpleoverlap(EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE)
        

        ##########################################
        # DEBUG: check bead overlap ##############
        if self.debug and check:
            if self.check_overlap():
                print(f'{EV_typeA=}')
                print(f'{EV_typeB=}')
                print(f'{EV_typeC=}')
                print(f'{EV_typeD=}')
                print(f'{EV_typeE=}')                
                sys.exit()    
        ##########################################
        return check   
    
    
    #########################################################################################
    ########## current largest distance #####################################################
    #########################################################################################
    
    # def init_nlargest(self):
    #     self.nlargest      = np.zeros(self.num_largest)
    #     self.bp_pos        = self.chain.conf[:,:3,3]
    #     self.bp_pos_backup = self.chain.backup_conf[:,:3,3]
    #     self.nlargest = update_largest_dist_list(self.get_evpos(),self.closed,self.nlargest)   
    
    # def largest_dist(self, intervals: List[List]) -> float:
    #     self.new_nlargest = np.copy(self.nlargest)
    #     self.new_nlargest = update_largest_dist_list(self.get_evpos(),self.closed,self.new_nlargest,intervals)
    #     largest = self.new_nlargest[0]
    #     print(f'largest = {largest}')
        
    #     check_nlargest = check_largest_dist_list(self.get_evpos(),self.closed,self.num_largest)
    #     # largest = check_nlargest[0]
        
    #     print('remove check')
    #     for i in range(self.num_largest):
    #         if np.abs(self.new_nlargest[i] - check_nlargest[i]) > 1e-14:
    #             print('nlargest discrepancy')
    #             print(f'{self.new_nlargest=}')
    #             print(f'{check_nlargest=}')
                
    #             for j in range(self.num_largest):
    #                 print(f'{j}: {self.new_nlargest[j] - check_nlargest[j]}')
                                        
    #             sys.exit()
    #     return largest
    
    
    def get_evpos(self, bp_pos: np.ndarray = None) -> np.ndarray:
        if bp_pos is None:
            return self.bp_pos[self.EV_beads] 
        return bp_pos[self.EV_beads]
                
    #########################################################################################
    ########## Check Intervals ##############################################################
    #########################################################################################
    
    def check_intervals(
        self,
        EV_typeA: List[int],
        EV_typeB: List[int],
        EV_typeC: List[int],
        EV_typeD: List[int],
        EV_typeE: List[int]
    ) -> bool:
                
        # Check EV_typeC
        for tC in range(len(EV_typeC)):
            # Check with all EV_typeA with singleMove check
            for tA in range(len(EV_typeA)-1,-1,-1):
                if (not self.check_interval_singleMove(EV_typeC[tC][self.EV_FROM],EV_typeC[tC][self.EV_TO],EV_typeA[tA][self.EV_FROM],EV_typeA[tA][self.EV_TO])):
                    return False

            # Check with all EV_typeB with doubleMove check
            for tB in range(0,len(EV_typeB)):
                if (not self.check_interval_doubleMove(EV_typeC[tC][self.EV_FROM],EV_typeC[tC][self.EV_TO],EV_typeB[tB][self.EV_FROM],EV_typeB[tB][self.EV_TO])):
                    return False

            # Check with all EV_typeD with doubleMove check
            for tD in range(0,len(EV_typeD)):
                if (not self.check_interval_doubleMove(EV_typeC[tC][self.EV_FROM],EV_typeC[tC][self.EV_TO],EV_typeD[tD][self.EV_FROM],EV_typeD[tD][self.EV_TO])):
                    return False

            # Check with all EV_typeE with doubleMove check
            for tE in range(0,len(EV_typeE)):
                if (not self.check_interval_doubleMove(EV_typeC[tC][self.EV_FROM],EV_typeC[tC][self.EV_TO],EV_typeE[tE][self.EV_FROM],EV_typeE[tE][self.EV_TO])):
                    return False

            # Check with all other EV_typeC
            for tC2 in range(tC+1,len(EV_typeC)):
                # A single interval can be split into two intervals when it crosses the periodic boundary. For typeC intervals
                # these intervals should not be mutually checked
                if ( EV_typeC[tC][self.EV_TYPE] != EV_typeC[tC2][self.EV_TYPE] ):
                    if (not self.check_interval_doubleMove(EV_typeC[tC][self.EV_FROM],EV_typeC[tC][self.EV_TO],EV_typeC[tC2][self.EV_FROM],EV_typeC[tC2][self.EV_TO])):
                        return False

        # Check EV_typeD
        for tD in range(0,len(EV_typeD)):
            # Check with all EV_typeA with singleMove check
            for tA in range(0,len(EV_typeA)):
                if (not self.check_interval_singleMove(EV_typeD[tD][self.EV_FROM],EV_typeD[tD][self.EV_TO],EV_typeA[tA][self.EV_FROM],EV_typeA[tA][self.EV_TO])):
                    return False

            # Check with all EV_typeB with doubleMove check
            for tB in range(0,len(EV_typeB)):
                if (not self.check_interval_doubleMove(EV_typeD[tD][self.EV_FROM],EV_typeD[tD][self.EV_TO],EV_typeB[tB][self.EV_FROM],EV_typeB[tB][self.EV_TO])):
                    return False

            # Check with all EV_typeE with doubleMove check
            for tE in range(0,len(EV_typeE)):
                if (not self.check_interval_doubleMove(EV_typeD[tD][self.EV_FROM],EV_typeD[tD][self.EV_TO],EV_typeE[tE][self.EV_FROM],EV_typeE[tE][self.EV_TO])):
                    return False

            # Check with all other EV_typeD
            for tD2 in range(tD+1,len(EV_typeD)):
                # A single interval can be split into two intervals when it crosses the periodic boundary. For typeD intervals
                # these intervals SHOULD be mutally checked regardless!
                if (not self.check_interval_doubleMove(EV_typeD[tD][self.EV_FROM],EV_typeD[tD][self.EV_TO],EV_typeD[tD2][self.EV_FROM],EV_typeD[tD2][self.EV_TO])):
                    return False

            # Check within the interval itself
            if (not self.check_within_interval(EV_typeD[tD][self.EV_FROM],EV_typeD[tD][self.EV_TO])):
                return False

        # Check EV_typeB
        for tB in range(0,len(EV_typeB)):
            # Check with all EV_typeA with singleMove check
            for tA in range(0,len(EV_typeA)):
                if (not self.check_interval_singleMove(EV_typeB[tB][self.EV_FROM],EV_typeB[tB][self.EV_TO],EV_typeA[tA][self.EV_FROM],EV_typeA[tA][self.EV_TO])):
                    return False

            # Check with all EV_typeE with doubleMove check
            for tE in range(0,len(EV_typeE)):
                if (not self.check_interval_doubleMove(EV_typeB[tB][self.EV_FROM],EV_typeB[tB][self.EV_TO],EV_typeE[tE][self.EV_FROM],EV_typeE[tE][self.EV_TO])):
                    return False

        # Check additional Boundary pairs. These will always be checked regardless on whether the constituent monomers
        # are within one of the moved intervals.
        if (self.additional_boundcheck):
            for i in range(len(self.addboundpairs)):
                dist = doubleMove(self.addboundpairs[i][0],self.addboundpairs[i][1],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                if (dist < self.EV_dist):
                    return False
        return True
    
    #########################################################################################
    ########## Check Interval of Single Moves ###############################################
    #########################################################################################
    
    def check_interval_singleMove(self, A1: int, A2: int, B1: int, B2: int) -> bool:
        """
            Checks for projected overlap between EV beads in the two intervals limited by
            A1,A2 and B1,B2 respectively. This method assumes that the beads in the B
            interval have not moved, such that only the linear displacement of the beads
            in the A interval have to be considered. The checking is done by the
            singleMove method.
        """
        if self.debug:
            if (not (A2<B1 or B2 < A1)):
                print(f'Check Intervals {A1} {A2} {B1} {B2}')
                raise ValueError(f'ExVol.check_interval_singleMove(): Invalid Intervals! The invervals are not allowed to cross the periodic boundary!')

        elif self.numba_interval_checks:
            return singleMove_intervals(
                A1,A2,B1,B2,self.closed,
                self.bp_pos,self.bp_pos_backup,
                self.EV_dist,self.num_EV,self.EV_beads,
                self.curr_size_EV_bead,
                self.neighbour_skip_boundary_plus_one,
                self.neighbour_skip_plus_one)
        
        if (self.closed):      
        ##############################################################
        # PERIODIC BOUNDARY CONDITION

            # Left to right order: A - B
            if (A2<B1):
                a1 = max([A1,B2-self.num_EV+self.neighbour_skip_boundary_plus_one])
                a2 = min([A2,B1-self.neighbour_skip_plus_one])

                # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
                a = int(A1)
                while (a<=A2):
                    if ( a1<=a and a<=a2):
                        a=a2+1
                        continue
                    
                    b  = max ([B1,a+self.neighbour_skip_plus_one])
                    b2 = min ([B2,a+self.num_EV-self.neighbour_skip_boundary_plus_one])
                    while (b<=b2):
                        dist = singleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup)
                        if (dist < self.EV_dist):
                            # debug_plot(self.bp_pos,self.bp_pos_backup,self.EV_beads,a,b)
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
                    a += 1
            
            # Left to right order: B - A
            else:
                a1 = max([A1,B2+self.neighbour_skip_plus_one])
                a2 = min([A2,B1+self.num_EV-self.neighbour_skip_boundary_plus_one])
                # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
                a = int(A1)
                while (a<=A2):
                    if ( a1<=a and a<=a2):
                        a=a2+1
                        continue
                    
                    b  = max ([B1,a-self.num_EV+self.neighbour_skip_boundary_plus_one])
                    b2 = min ([B2,a-self.neighbour_skip_plus_one])
                    while (b<=b2):
                        dist = singleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup)
                        if (dist < self.EV_dist):
                            # debug_plot(self.bp_pos,self.bp_pos_backup,self.EV_beads,a,b)
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
                    a+=1
        else:
        ##############################################################
        # NON-PERIODIC BOUNDARY CONDITION
        
            # Left to right order: A - B
            if (A2<B1):
                # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
                # In this case this region is only at the right boundary of the interval A.
                a1 = int(A1)
                a2 = min([A2,B1-self.neighbour_skip_plus_one])
                
                # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
                for a in range(a2+1,A2+1):
                    b = max([B1,a+self.neighbour_skip_plus_one])
                    while (b<=B2):
                        dist = singleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup)
                        if (dist < self.EV_dist):
                            # debug_plot(self.bp_pos,self.bp_pos_backup,self.EV_beads,a,b)
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
            
            # Left to right order: B - A
            else:
                # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
                # In this case this region is only at the left boundary of the interval A.
                a1 = max([A1,B2+self.neighbour_skip_plus_one])
                a2 = int(A2)
                
                # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
                for a in range(A1,a1):      
                    b  = int(B1)
                    b2 = min([B2,a-self.neighbour_skip_plus_one])
                    while (b<=b2):
                        dist = singleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup)
                        if (dist < self.EV_dist):
                            # debug_plot(self.bp_pos,self.bp_pos_backup,self.EV_beads,a,b)
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
                        
        ##############################################################
        # Finally check all pairs outside the range of potential neighbour skips. I.e. the bulk of the intervals.
        for a in range(a1,a2+1):
            b = int(B1)
            while (b<=B2):
                dist = singleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup)
                if (dist < self.EV_dist):
                    # debug_plot(self.bp_pos,self.bp_pos_backup,self.EV_beads,a,b)
                    return False
                #b+=1
                b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
        return True
        
    
    #########################################################################################
    ########## Check Interval of Double Moves ###############################################
    #########################################################################################
    
    def check_interval_doubleMove(self, A1: int, A2: int, B1: int, B2: int) -> bool:
        """
            Checks for projected overlap between EV beads in the two intervals limited by
            A1,A2 and B1,B2 respectively. This method assumes that the beads in both
            intervals have not moved such that The checking needs to be done by the 
            doubleMove method.
        """

        if self.debug:
            if (not (A2<B1 or B2 < A1)):
                print(f'Check Intervals {A1} {A2} {B1} {B2}')
                raise ValueError(f'ExVol.check_interval_doubleMove(): Invalid Intervals! The invervals are not allowed to cross the periodic boundary!')

        elif self.numba_interval_checks:
            return doubleMove_intervals(
                A1,A2,B1,B2,self.closed,
                self.bp_pos,self.bp_pos_backup,
                self.EV_dist,self.num_EV,self.EV_beads,
                self.curr_size_EV_bead,
                self.neighbour_skip_boundary_plus_one,
                self.neighbour_skip_plus_one)
        
        if (self.closed):           
        ##############################################################
        # PERIODIC BOUNDARY CONDITION

            # Left to right order: A - B
            if (A2<B1):
                a1 = max([A1,B2-self.num_EV+self.neighbour_skip_boundary_plus_one])
                a2 = min([A2,B1-self.neighbour_skip_plus_one])

                # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
                a = int(A1)
                while (a<=A2):
                    if ( a1<=a and a<=a2):
                        a=a2+1
                        continue
                    
                    b  = max ([B1,a+self.neighbour_skip_plus_one])
                    b2 = min ([B2,a+self.num_EV-self.neighbour_skip_boundary_plus_one])
                    while (b<=b2):
                        dist = doubleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                        if (dist < self.EV_dist):
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
                    a += 1
            
            # Left to right order: B - A
            else:
                a1 = max([A1,B2+self.neighbour_skip_plus_one])
                a2 = min([A2,B1+self.num_EV-self.neighbour_skip_boundary_plus_one])
                # Check for overlaps within the boundary region of A and region B. Pairs within neighbour skip region are omitted.
                a = int(A1)
                while (a<=A2):
                    if ( a1<=a and a<=a2):
                        a=a2+1
                        continue
                    
                    b  = max ([B1,a-self.num_EV+self.neighbour_skip_boundary_plus_one])
                    b2 = min ([B2,a-self.neighbour_skip_plus_one])
                    while (b<=b2):
                        dist = doubleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                        if (dist < self.EV_dist):
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
                    a+=1
        else:
        ##############################################################
        # NON-PERIODIC BOUNDARY CONDITION
        
            # Left to right order: A - B
            if (A2<B1):
                # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
                # In this case this region is only at the right boundary of the interval A.
                a1 = int(A1)
                a2 = min([A2,B1-self.neighbour_skip_plus_one])
                
                # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
                for a in range(a2+1,A2+1):
                    b = max([B1,a+self.neighbour_skip_plus_one])
                    while (b<=B2):
                        dist = doubleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                        if (dist < self.EV_dist):
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
            
            # Left to right order: B - A
            else:
                # Define the boundary regions in which pairchecking potentially has to be omitted due to proximity along the chain.
                # In this case this region is only at the left boundary of the interval A.
                a1 = max([A1,B2+self.neighbour_skip_plus_one])
                a2 = int(A2)
                
                # Check for overlaps within this boundary region of A and region B. Pairs within neighbour skip region are omitted
                for a in range(A1,a1):                
                    b  = int(B1)
                    b2 = min([B2,a-self.neighbour_skip_plus_one])
                    while (b<=b2):
                        dist = doubleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                        if (dist < self.EV_dist):
                            return False
                        #b+=1
                        b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
                        
        ##############################################################
        # Finally check all pairs outside the range of potential neighbour skips. I.e. the bulk of the intervals.
        for a in range(a1,a2+1):
            b = int(B1)
            while (b<=B2):
                dist = doubleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                if (dist < self.EV_dist):
                    return False
                #b+=1
                b += int((dist-self.EV_dist)//self.curr_size_EV_bead+1)
        return True
    
    #########################################################################################
    ########## Debugging Methods ############################################################
    #########################################################################################

    def check_overlap(self):
        # if self.closed:
        #     raise ValueError(f'check_overlap: proper overlap check not implemented for closed molecules')
                
        if self.quick_overlap_check:
            return check_overlap_numba(
                self.bp_pos,
                self.EV_beads,
                self.EV_dist,
                self.closed,
                self.neighbour_skip,
                self.additional_boundcheck,
                self.addboundpairs
            )
        
        overlap = False
        if not self.closed:
            for a in range(self.num_EV-1):
                for b in range(a+self.neighbour_skip+1,self.num_EV):
                    p1 = self.bp_pos[self.EV_beads[a]]
                    p2 = self.bp_pos[self.EV_beads[b]]
                    dist = np.linalg.norm(p1-p2)
                    if (dist < self.EV_dist):
                        print(f'Overlap: {dist} ({self.EV_dist}) -> {a} {b} - {self.EV_beads[a]} {self.EV_beads[b]}')
                        overlap = True 
        
        else:
            overlap = False
            a1 = self.neighbour_skip
            if self.additional_boundcheck:
                a1 += 1
            
            for a in range(a1,self.num_EV):
                b = a+self.neighbour_skip+1
                while b<self.num_EV:
                    p1 = self.bp_pos[self.EV_beads[a]]
                    p2 = self.bp_pos[self.EV_beads[b]]

                    dist = np.linalg.norm(p1-p2)
                    if (dist < self.EV_dist):
                        print(f'Overlap: {dist} ({self.EV_dist}) -> {a} {b} - {self.EV_beads[a]} {self.EV_beads[b]}')
                        overlap = True
                    b+=1

            # The first elements for which there are neighbour skips.
            ns=self.neighbour_skip
            if self.additional_boundcheck:
                ns+=1
            
            for a in range(0,a1):  
                b2 = (a-ns-1)%self.num_EV
                b = a+self.neighbour_skip+1
                while (b<=b2):
                    p1 = self.bp_pos[self.EV_beads[a]]
                    p2 = self.bp_pos[self.EV_beads[b]]

                    dist = np.linalg.norm(p1-p2)
                    if (dist < self.EV_dist):
                        print(f'Overlap: {dist} ({self.EV_dist}) -> {a} {b} - {self.EV_beads[a]} {self.EV_beads[b]}')
                        overlap = True
                    b+=1

            if self.additional_boundcheck:
                for i in range(0,len(self.addboundpairs)):
                    p1 = self.bp_pos[self.addboundpairs[i][0]]
                    p2 = self.bp_pos[self.addboundpairs[i][1]]
                    
                    dist = np.linalg.norm(p1-p2)
                    if (dist < self.EV_dist):
                        print(f'Overlap: {dist} ({self.EV_dist}) -> {a} {b} - {self.EV_beads[a]} {self.EV_beads[b]}')
                        print(f'special check')
                        overlap = True
                        
        return overlap   
    
     
@cond_jit
def check_overlap_numba(
    bp_pos: np.ndarray,
    EV_beads: np.ndarray,
    EV_dist: float,
    closed: bool,
    neighbour_skip: int,
    additional_boundcheck: bool,
    addboundpairs: np.ndarray
    ):
    # if self.closed:
    #     raise ValueError(f'check_overlap: proper overlap check not implemented for closed molecules')
    
    num_EV = len(EV_beads)
    overlap = False
    if not closed:
        for a in range(num_EV-1):
            for b in range(a+neighbour_skip+1,num_EV):
                p1 = bp_pos[EV_beads[a]]
                p2 = bp_pos[EV_beads[b]]
                dist = np.linalg.norm(p1-p2)
                if (dist < EV_dist):
                    overlap = True 
    else:
        overlap = False
        a1 = neighbour_skip
        if additional_boundcheck:
            a1 += 1
        for a in range(a1,num_EV):
            b = a+neighbour_skip+1
            while b<num_EV:
                p1 = bp_pos[EV_beads[a]]
                p2 = bp_pos[EV_beads[b]]
                dist = np.linalg.norm(p1-p2)
                if (dist < EV_dist):
                    overlap = True
                b+=1

        # The first elements for which there are neighbour skips.
        ns=neighbour_skip
        if additional_boundcheck:
            ns+=1
        
        for a in range(0,a1):  
            b2 = (a-ns-1)%num_EV
            b = a+neighbour_skip+1
            while (b<=b2):
                p1 = bp_pos[EV_beads[a]]
                p2 = bp_pos[EV_beads[b]]
                dist = np.linalg.norm(p1-p2)
                if (dist < EV_dist):
                    overlap = True
                b+=1

        if additional_boundcheck:
            for i in range(0,len(addboundpairs)):
                p1 = bp_pos[int(addboundpairs[i,0])]
                p2 = bp_pos[int(addboundpairs[i,1])]
                
                dist = np.linalg.norm(p1-p2)
                if (dist < EV_dist):
                    overlap = True                  
    return overlap    
