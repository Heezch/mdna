import os
import sys
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from ..._jit import cond_jit
from .ExVol import ExVol
from ..chain import Chain


class EVBeads(ExVol):
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
        print(f'{self.neighbour_skip}')
        print(f'{self.num_bp*self.num_bp_per_EV}')

        self.additional_boundcheck=False
        if (self.num_bp%self.num_bp_per_EV!=0 and chain.closed):
            print('mismatch')
            self.additional_boundcheck=True
            self.addboundpairs = []
            for i in range(self.neighbour_skip+1):
                self.addboundpairs.append([i*self.num_bp_per_EV,self.num_bp-(self.neighbour_skip+1-i)*self.num_bp_per_EV])
            self.neighbour_skip_boundary += 1
            self.neighbour_skip_boundary_plus_one += 1
        
        print('')
        print("######################################")
        print("#### INITIALIZING EXCLUDED VOLUME ####")
        print("######################################")
        print(f" Excluded Volume Beads: ")
        print(f"   number of EV beads: {self.num_EV}")
        print(f"   bp per EV bead:     {self.num_bp_per_EV}")
        print(f"   Effective size:     {self.eff_size_EV_bead}")
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
        
        # self.bp_pos = chain.conf[:,:3,3]
        # if self.EV_dist > 0 and self.check_overlap():
        #     raise ValueError(f'EVBeads: Overlap detected in intial configuration!')


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
        """
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
        
        return EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE

        # #ifdef DEBUG_EXVOL
        # // test consistency
        # for (int i=1;i<EV_intervals.size();i++) {
        #     if (EV_intervals[i][self.EV_FROM] != EV_intervals[i-1][self.EV_TO]+1) {
        #         std::cout << "intervals inconsistent!"  << std::endl;
        #         for (int j=0;j<EV_intervals.size();j++) {
        #             std::cout << EV_intervals[j].t();
        #         }
        #         std::cout << std::endl;
        #         for (int i=0;i<moved->size();i++) {
        #             std::cout << (*moved)[i].t();
        #         }
        #         break;
        #     }
        # }
        # if (EV_intervals[EV_intervals.size()-1](1) != num_EV-1) {
        #     std::cout << "intervals inconsistent! (last)" << std::endl;
        #     std::cout << "Index of last EV bead: " << num_EV-1 << std::endl;
        #     for (int j=0;j<EV_intervals.size();j++) {
        #         std::cout << EV_intervals[j].t();
        #     }
        #     std::cout << std::endl;
        #     for (int i=0;i<moved->size();i++) {
        #         std::cout << (*moved)[i].t();
        #     }
        # }
        # #endif
        
    #########################################################################################
    ########## Main Check ###################################################################
    #########################################################################################
    
    def check(self, moved: List = None):
        
        # print('#####################################')
        # print('moved:')
        # print(moved)
        self.counter += 1
        EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE = self.cal_EV_intervals(moved)
        # check = True

        # print(f'{EV_typeA=}')
        # print(f'{EV_typeB=}')
        # print(f'{EV_typeC=}')
        # print(f'{EV_typeD=}')
        # print(f'{EV_typeE=}')
        
        self.bp_pos        = self.chain.conf[:,:3,3]
        self.bp_pos_backup = self.chain.backup_conf[:,:3,3]
        
        if (self.check_crossings):
            check = self.check_intervals(EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE)
        else:
            check = self.check_intervals_simpleoverlap(EV_typeA,EV_typeB,EV_typeC,EV_typeD,EV_typeE)
        
        # if not check:
        #     print('#####################')
        #     print('rejected')
        #     print(f'{EV_typeA=}')
        #     print(f'{EV_typeB=}')
        #     print(f'{EV_typeC=}')
        #     print(f'{EV_typeD=}')
        #     print(f'{EV_typeE=}')
        #     sys.exit()
        
        if check:
            if self.check_overlap():
                sys.exit()    
            # else:
            #     print('checks out')
            
        return check   
    
    
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
        
        # print('################')
        # print(EV_typeA)
        # print(EV_typeB)
        # print(EV_typeC)
        # print(EV_typeD)
        # print(EV_typeE)
        
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
        
        # #ifdef DEBUG_EXVOL
        # if (!(A2<B1 || B2 < A1)) {
        #     std::cout << "Check Intervals " << A1 << " " << A2 << " " << B1 << " " << B2 << std::endl;
        #     throw std::invalid_argument("ExVol::check_interval_singleMove(): Invalid Intervals! The invervals are not allowed to cross the periodic boundary!");
        # }
        # #endif

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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
            
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
                        
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
                b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
            
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
                        b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
                        
        ##############################################################
        # Finally check all pairs outside the range of potential neighbour skips. I.e. the bulk of the intervals.
        for a in range(a1,a2+1):
            b = int(B1)
            while (b<=B2):
                dist = doubleMove(self.EV_beads[a],self.EV_beads[b],self.bp_pos,self.bp_pos_backup,self.EV_dist)
                if (dist < self.EV_dist):
                    return False
                #b+=1
                b += int((dist-self.EV_dist)//self.eff_size_EV_bead+1)
        return True
    
    #########################################################################################
    ########## Debugging Methods ############################################################
    #########################################################################################

    def check_overlap(self):
        # if self.closed:
        #     raise ValueError(f'check_overlap: proper overlap check not implemented for closed molecules')
        
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
    
    
#########################################################################################
########## Double Move ##################################################################
#########################################################################################

@cond_jit
def doubleMove(id1: int, id2: int, bp_pos: np.ndarray, bp_pos_backup: np.ndarray, EV_dist: float) -> float:
    p1  = bp_pos_backup[id1]
    p1p = bp_pos[id1]
    p2  = bp_pos_backup[id2]
    p2p = bp_pos[id2]
    
    dist_primes = np.linalg.norm(p2p-p1p)
    if (dist_primes<EV_dist):
        return dist_primes

    Delta_p = p1-p2
    Delta_v = p1p-p2p-Delta_p

    # check if both were translated by the same vector
    dist = np.dot(Delta_v,Delta_v)
    if (dist < 1e-10):
        return dist_primes
    
    lamb = -np.dot(Delta_p,Delta_v)/dist
    # check if the closest approach is outside the moved interval
    if (lamb < 0):
        return np.linalg.norm(Delta_p)
    
    if (lamb > 1):
        return dist_primes
    
    dvec = Delta_p+lamb*Delta_v
    dist = np.linalg.norm(dvec)
    return dist


#########################################################################################
########## Single Move ##################################################################
#########################################################################################

@cond_jit
def singleMove(id1: int, id2: int, bp_pos: np.ndarray, bp_pos_backup: np.ndarray)-> float:
    p1  = bp_pos_backup[id1]
    p1p = bp_pos[id1]
    p2  = bp_pos[id2]
    
    v = p1p-p1
    nv = np.linalg.norm(v)
    w1 = p2-p1
    if (nv<1e-12):
        return np.linalg.norm(p2-p1)
    v = v/nv
    w2 = p1p-p2
    n_com = np.dot(v,w1)
    if (n_com < 0):
        return np.linalg.norm(p2-p1)
    if (np.dot(v,w2) < 0):
        return np.linalg.norm(p2-p1p)
    return np.linalg.norm( w1-n_com*v )

#########################################################################################
########## Plotting for Debug ###########################################################
#########################################################################################

import matplotlib.pyplot as plt
def debug_plot(pos, backup, EV_beads, a, b):
    debug = False
    if not debug: return
    
    print('####################')
    print(f'overlap {a} - {b}')
    ida = EV_beads[a]
    idb = EV_beads[b]
    print(f'bp ids: {ida} - {idb}')
    print(pos[ida])
    print(pos[idb])
    print(np.linalg.norm(pos[ida]-pos[idb]))
        
        
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax = plt.gca(projection='3d')
    
    ida = EV_beads[a]
    idb = EV_beads[b]
    
    segpos = pos[ida:idb+1]
    rempos = np.array([pos[i] for i in range(len(pos)) if i not in np.arange(ida,idb+1)])
    
    segpos_backup = backup[ida:idb+1]
    rempos_backup = np.array([backup[i] for i in range(len(pos)) if i not in np.arange(ida,idb+1)])
    
    ax.plot(*backup.T,color='black',label='Control Points',lw=1)
    ax.scatter(*rempos_backup.T,color='green',s=16,edgecolor='black',alpha=0.2,zorder=2)
    ax.scatter(*segpos_backup.T,color='red',s=16,edgecolor='black',zorder=3,alpha=0.5)
    evpos_backup = np.array([backup[id] for id in EV_beads])
    ax.scatter(*evpos_backup.T,color='blue',s=500,edgecolor='black',alpha=0.2)
    
    ax.plot(*pos.T,color='black',label='Control Points',lw=1)
    ax.scatter(*rempos.T,color='green',s=16,edgecolor='black',alpha=0.4,zorder=2)
    ax.scatter(*segpos.T,color='red',s=16,edgecolor='black',zorder=3)
    
    evpos = np.array([pos[id] for id in EV_beads])
    ax.scatter(*evpos.T,color='blue',s=500,edgecolor='black',alpha=0.4)
    
    
    com = np.mean(pos,axis=0)
    maxrge = np.max([np.max(pos[:,i])-np.min(pos[:,i]) for i in range(3)])
    margin = maxrge*0.01
    halfrge = maxrge*0.5+margin
    rges = []
    for i in range(3):
        rges.append([com[i]-halfrge,com[i]+halfrge])
    for x in range(2):
        for y in range(2):
            for z in range(2):
                ax.scatter([rges[0][x]],[rges[1][y]],[rges[2][z]],alpha=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()