import numpy as np
from qiskit import QuantumCircuit
from dataclasses import dataclass
from enum import Enum

class AttackType(Enum):
    NONE = "none"
    INTERCEPT_RESEND = "intercept_resend"
    PHOTON_SPLITTING = "photon_splitting"
    BEAM_SPLITTING = "beam_splitting"

@dataclass
class ChannelMetrics:
    transmittance: float
    noise: float
    qber_est: float
    attack_type: AttackType = AttackType.NONE
    attack_strength: float = 0.0

class EavesdropperSimulator:
    """Simulates various eavesdropping attacks on the quantum channel"""
    
    def __init__(self, attack_type: AttackType, strength: float = 0.5):
        self.attack_type = attack_type
        self.strength = strength

    def apply_attack(self, qc: QuantumCircuit, basis: int, bit: int) -> QuantumCircuit:
        """Apply eavesdropping attack to quantum circuit"""
        attacked_qc = qc.copy()
        
        if self.attack_type == AttackType.INTERCEPT_RESEND:
            if np.random.rand() < self.strength:
                eve_basis = np.random.randint(2)
                if eve_basis == 1:
                    attacked_qc.h(0)
                attacked_qc.measure_all()
                
                eve_bit = np.random.randint(2)
                attacked_qc = QuantumCircuit(1, 1)
                if eve_basis == 0:
                    if eve_bit == 1:
                        attacked_qc.x(0)
                else:
                    if eve_bit == 0:
                        attacked_qc.h(0)
                    else:
                        attacked_qc.x(0)
                        attacked_qc.h(0)
        
        elif self.attack_type == AttackType.PHOTON_SPLITTING:
            if np.random.rand() < self.strength * 0.3:
                attacked_qc.rz(np.pi * np.random.rand(), 0)
        
        elif self.attack_type == AttackType.BEAM_SPLITTING:
            if np.random.rand() < self.strength * 0.4:
                attacked_qc.reset(0)
                if np.random.rand() < 0.5:
                    attacked_qc.x(0)
        
        return attacked_qc
