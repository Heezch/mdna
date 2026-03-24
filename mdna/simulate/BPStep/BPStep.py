import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import numpy.typing as npt

from ..chain import Chain
from ... import math as so3


class BPStep(ABC):
    
    def __init__(
        self,
        chain: Chain,
        sequence: str,
        closed: bool = False,
        static_group: bool = True,
        temp: float = 300,
        conserve_twist: bool = True
    ):
        self.chain = chain
        self.sequence = sequence
        self.closed = closed
        self.static_group = static_group
        self.temp = 300
        self.conserve_twist = conserve_twist
        self.twist_conserved = True

        self.nbp = len(chain.conf)
        if closed:
            self.nbps = self.nbp
        else:
            self.nbps = self.nbp - 1
        
        self.move_pending = False
        self.current_deforms = np.zeros((self.nbps, 6))
        self.proposed_deforms = np.zeros((self.nbps, 6))
        self.proposed_ids = []

        # # currently not in use
        # self.max_dist_set = False
        # self.max_dist = 0
        
        # termini tracing
        self.trace_termini_active = False
        self.first_triad_changed = False
        self.last_triad_changed  = False
        
        # angle tracing flags
        self.angle_tracing_last_active = False
        self.angle_tracing_first_active = False
        
        # stretching force
        self.stretching_force_active = False
        self.beta_force_vector = None


        # inits parameters and energies
        self.init_finished = False # needed for certain check flags that should not trigger an exception upon initation
        self._init_params()
        self.init_static()
        self.set_temperature(temp)
        # self.init_conf() # -> init_conf is called in set_temperature
        self.init_finished = True

    #########################################################################################
    #########################################################################################

    def init_conf(self) -> float:
        for id in range(self.nbps):
            self.propose_move(
                id, self.chain.conf[id], self.chain.conf[(id + 1) % self.nbp]
            )
        dE = self.eval_delta_E()
        self.set_move(True)
        return dE

    #########################################################################################
    #########################################################################################

    def propose_move(self, id: int, triad1: np.ndarray, triad2: np.ndarray) -> None:
        self.move_pending = True
        
        # calculate deformation
        if self.static_group:
            X = so3.se3_rotmat2euler(
                self.gs_mats_inv[id] @ so3.se3_triads2rotmat(triad1, triad2)
            )
        else:
            X = (
                so3.se3_rotmat2euler(so3.se3_triads2rotmat(triad1, triad2))
                - self.gs_vecs[id]
            )
        self.proposed_deforms[id] = X
        self.proposed_ids.append(id)
        
        # check conservation of twist
        if self.conserve_twist:
            if np.abs(self.proposed_deforms[id,2]-self.current_deforms[id,2]) > 1.5707963265 and self.init_finished:
                self.twist_conserved = False
            else:
                self.twist_conserved = True
        
        # check if termini changed
        if self.trace_termini_active:
            # first
            if id == 0:
                self.propose_move_first(triad1)
            # last
            if id == self.nbp-2:
                self.propose_move_last(triad2)                
            # if closed and change is across boundary
            if self.closed and id == self.nbp-1:
                self.propose_move_first(triad2)
                self.propose_move_last(triad1)
    
    #########################################################################################
    #########################################################################################
    # evaluate the change in energy (in units of kT)

    def eval_delta_E(self) -> float:
        """
            Change in elastic energy in kT
        """
        
        if self.conserve_twist and not self.twist_conserved:
            # print('twist not conserved')
            return 1e10
        
        dE = self._eval_delta_E()
        dE += self.eval_delta_E_stretching_force()
        return dE
    
    #########################################################################################
    #########################################################################################
    # retrieves the total current energy (handled by this object). Elastic and work due to external forces and torques

    def get_total_energy(self) -> float:
        """ 
            Retrieves total current energy in units of kT. This method does not recalculate the current state, but pulls 
            current energies from memory.
        """
        E = self._get_total_energy()
        E += self.eval_current_E_stretching_force()
        return E

    #########################################################################################
    #########################################################################################
    # Calculate the total current energy (handled by this object). Elastic and work due to external forces and torques

    def recalculate_total_energy(self) -> float:
        self.init_conf()
        return self.get_total_energy()

    #########################################################################################
    #########################################################################################

    def set_move(self, accept: bool = True) -> None:
        if accept:
            self.current_deforms = np.copy(self.proposed_deforms)
        else:
            self.proposed_deforms = np.copy(self.current_deforms)
        self.set_energy(accept)
        self.proposed_ids = []
        
        # set changes to first and last triad
        if self.first_triad_changed:
            self._set_move_first(accept)
        if self.last_triad_changed:
            self._set_move_last(accept) 
                   
        self.move_pending = False
                      
    #########################################################################################
    #########################################################################################

    def init_static(self) -> None:
        if self.static_group:
            self.gs_mats = np.zeros((len(self.gs_vecs), 4, 4))
            self.gs_mats_inv = np.zeros((len(self.gs_vecs), 4, 4))
            for i, gs in enumerate(self.gs_vecs):
                self.gs_mats[i] = so3.se3_euler2rotmat(gs)
                self.gs_mats_inv[i] = so3.se3_inverse(self.gs_mats[i])

    #########################################################################################
    #########################################################################################
    
    def set_temperature(self, temp: float) -> None:
        self._set_temperature(temp)
        self._change_stretching_force_temperature(temp)
        self.temp = temp
    
    #########################################################################################
    #########################################################################################

    def check_deform_consistency(self) -> None:
        consistent = True
        for id in range(self.chain.nbp - 1):
            triad1 = self.chain.conf[id]
            triad2 = self.chain.conf[id + 1]
            if self.static_group:
                X = so3.se3_rotmat2euler(
                    self.gs_mats_inv[id] @ so3.se3_triads2rotmat(triad1, triad2)
                )
            else:
                X = (
                    so3.se3_rotmat2euler(so3.se3_triads2rotmat(triad1, triad2))
                    - self.gs_vecs[id]
                )

            if np.abs(np.sum(self.current_deforms[id] - X)) > 1e-8:
                np.set_printoptions(linewidth=250, precision=16, suppress=True)
                print("Inconsistent deformation:")
                print(f" id:         {id}")
                print(f" stored:     {self.current_deforms[id]}")
                print(f" calculated: {X}")
                consistent = False
        return consistent

    #########################################################################################
    #### Tracing of Termini #################################################################
    #########################################################################################
    # termini changes may be declared without triggering an energy evaluation
    # 
    #  ->   This may be pertinent if the respective terminal triad was moved in he context 
    #       of a cluster translation or rotation which left the relative orientation of these
    #       triads with their respective neighbors unchanged. Such a move could also be 
    #       declared via propose_move, but this would trigger unnecessary energy evaluation. 
    #       The only result would be some unnecessary computation. The behavior is equivalent 
    #       for both options.
    
    def propose_move_first(self, first_triad: np.ndarray) -> None:
        if not self.trace_termini_active:
            return
        # self.termini_changed = True
        self.first_triad_changed = True
        self.proposed_first_triad = np.copy(first_triad)
        self._propose_angle_first()
        
    def propose_move_last(self, last_triad: np.ndarray) -> None:
        if not self.trace_termini_active:
            return
        # self.termini_changed = True
        self.last_triad_changed = True
        self.proposed_last_triad = np.copy(last_triad)
        self._propose_angle_last()
        
    def _set_move_first(self,accept: bool=True) -> None:
        if not self.first_triad_changed:
            return
        if accept:
            self.current_first_triad = np.copy(self.proposed_first_triad)
        else:
            self.proposed_first_triad = np.copy(self.current_first_triad)
        self._set_move_angle_first(accept)
            
        self.first_triad_changed = False 
        
    def _set_move_last(self,accept: bool=True) -> None:
        if not self.last_triad_changed:
            return
        if accept:
            self.current_last_triad = np.copy(self.proposed_last_triad)
        else:
            self.proposed_last_triad = np.copy(self.current_last_triad)
        self._set_move_angle_last(accept)
        self.last_triad_changed = False 
    
    def trace_termini(self) -> None:
        self.trace_termini_active = True
        self.current_first_triad = np.copy(self.chain.conf[0])
        self.current_last_triad  = np.copy(self.chain.conf[-1])
        self.proposed_first_triad = np.copy(self.chain.conf[0])
        self.proposed_last_triad  = np.copy(self.chain.conf[-1])
        # self.termini_changed = False
        self.first_triad_changed = False
        self.last_triad_changed = False
        
    #########################################################################################
    #########################################################################################
    # Tracing rotation angle of first and last triad
    

    #####################
    # handle last angle #

    def trace_angle_last_triad(self, set_tracing: bool=True) -> None:
        self.angle_tracing_last_active = set_tracing
        if not set_tracing: 
            return
        if not self.trace_termini_active:
            self.trace_termini()
        self.angle_tracing_last_ref_triad = np.copy(self.chain.conf[-1])
        self.angle_tracing_last_current_angle  = 0
        self.angle_tracing_last_proposed_angle = 0

    def _propose_angle_last(self) -> float:
        if not self.angle_tracing_last_active: 
            return
        
        # this could be improved!
        # check via dot product of last triad vector
        # then only consider 2x2 block of self.current_last_triad[:3,:3].T @ self.proposed_last_triad[:3,:3]
        # to calculate the angle
        
        dOm = so3.rotmat2euler(self.current_last_triad[:3,:3].T @ self.proposed_last_triad[:3,:3])
        # kill if tangent has changed
        if np.abs(dOm[0]) > 1e-10 or np.abs(dOm[1]) > 1e-10:
            raise ValueError(f'BPStep._propose_angle_last: tangent of last triad has changed, while tracing the rotation angle (specified by the rotation around the tangent). Perhaps you are using a move that does not preserve the orientation of the last tangent.')
        self.angle_tracing_last_proposed_angle = self.angle_tracing_last_current_angle + dOm[2]
        return dOm[2]
    
    def _set_move_angle_last(self,accept: bool) -> None:
        if not self.angle_tracing_last_active:
            return
        if accept:
            self.angle_tracing_last_current_angle = self.angle_tracing_last_proposed_angle
        else:
            self.angle_tracing_last_proposed_angle = self.angle_tracing_last_current_angle
        
    def get_angle_last(self):
        if not self.angle_tracing_last_active:
            return 0
        return self.angle_tracing_last_current_angle
    
    #####################
    # handle first angle #
    
    def trace_angle_first_triad(self, set_tracing: bool=True) -> None:
        self.angle_tracing_first_active = set_tracing
        if not set_tracing: 
            return
        if not self.trace_termini_active:
            self.trace_termini()
        self.angle_tracing_first_ref_triad = np.copy(self.chain.conf[0])
        self.angle_tracing_first_current_angle  = 0
        self.angle_tracing_first_proposed_angle = 0

    def _propose_angle_first(self) -> float:
        if not self.angle_tracing_first_active: 
            return        
        dOm = so3.rotmat2euler(self.current_first_triad[:3,:3].T @ self.proposed_first_triad[:3,:3])
        # kill if tangent has changed
        if np.abs(dOm[0]) > 1e-10 or np.abs(dOm[1]) > 1e-10:
            raise ValueError(f'BPStep._propose_angle_first: tangent of first triad has changed, while tracing the rotation angle (specified by the rotation around the tangent). Perhaps you are using a move that does not preserve the orientation of the first tangent.')
        self.angle_tracing_first_proposed_angle = self.angle_tracing_first_current_angle + dOm[2]
        return dOm[2]
    
    def _set_move_angle_first(self,accept: bool) -> None:
        if not self.angle_tracing_first_active:
            return
        if accept:
            self.angle_tracing_first_current_angle = self.angle_tracing_first_proposed_angle
        else:
            self.angle_tracing_first_proposed_angle = self.angle_tracing_first_current_angle

    def get_angle_first(self):
        if not self.angle_tracing_first_active:
            return 0
        return self.angle_tracing_first_current_angle
    
    #########################################################################################
    #########################################################################################
    # Linear Force
    
    def set_stretching_force(self, beta_force_vector: np.ndarray) -> None:
        """
            Activate stretching force. The vector beta_force_vector is assumed to be 
            $\beta \vec{F}$ according the the current temperature of this object.
        """
        if self.closed:
            return
        self.stretching_force_active = True
        self.beta_force_vector = np.copy(beta_force_vector)
        self.trace_termini()
    
    def _change_stretching_force_temperature(self, temp: float) -> None:
        if not self.stretching_force_active:
            return
        self.beta_force_vector = self.beta_force_vector * self.temp / temp

    def eval_delta_E_stretching_force(self) -> float:
        # if not self.termini_changed or not self.stretching_force_active:
        if (not self.first_triad_changed and not self.last_triad_changed) \
            or not self.stretching_force_active:
            return 0
        
        Rnew = self.proposed_last_triad[:3,3] - self.proposed_first_triad[:3,3]
        Rold = self.current_last_triad[:3,3]  - self.current_first_triad[:3,3]
        return np.dot(Rold,self.beta_force_vector) - np.dot(Rnew,self.beta_force_vector)
    
    def eval_current_E_stretching_force(self) -> float:
        if not self.stretching_force_active:
            return 0
        R = self.current_last_triad[:3,3]  - self.current_first_triad[:3,3]
        return -np.dot(R,self.beta_force_vector)   
    
    
    #########################################################################################
    #########################################################################################
    
    @abstractmethod
    def _init_params(self) -> None:
        """Initialte all parameters necessary for energy evaluation.
        The following variables also need to be set here:
            self.gs_vecs (groundstate vectors)
        """
        pass

    @abstractmethod
    def _eval_delta_E(self) -> float:
        pass

    @abstractmethod
    def set_energy(self, accept: bool = True) -> None:
        """
            TODO: include calculation of torque and force related work in the stored energies 
            that will be set or rejected with this method. 
        """
        pass

    @abstractmethod
    def _get_total_energy(self) -> float:
        pass

    @abstractmethod
    def _set_temperature(self, new_temp: float) -> None:
        # self.temp does not need to be set in this method (done in self.set_temperature)
        pass

    #########################################################################################
    #########################################################################################
