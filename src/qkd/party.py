import numpy as np
import hashlib
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from typing import Optional

from .device import ResourceLimitedDevice
from .channel import ChannelMetrics, EavesdropperSimulator
from .error_correction import EnhancedErrorCorrection, ErrorCorrectionType

class PerformanceMetrics:
    def __init__(self):
        self.key_rate_bps = 0
        self.qber = 0
        self.ram_usage_kb = 0
        self.cpu_cycles = 0
        self.power_mw = 0
        self.success_rate = 0
        self.final_key_length = 0
        self.raw_key_length = 0

class EnhancedQKDParty:
    """Enhanced QKD party with attack detection and multiple error correction"""
    
    def __init__(self, name: str, device: ResourceLimitedDevice, block_size: int, 
                 error_correction: ErrorCorrectionType = ErrorCorrectionType.LDPC):
        self.name = name
        self.device = device
        self.block_size = block_size
        self.error_corrector = EnhancedErrorCorrection(error_correction)
        self.backend = Aer.get_backend('qasm_simulator')
        
        self.raw_bits = None
        self.bases = None
        self.measured_bits = None
        self.sifted_key = None
        self.final_key = None
        
        self.metrics = PerformanceMetrics()
    
    def generate_random_data(self) -> bool:
        """Generate random bits and bases"""
        ram_needed = self.block_size * 3 * 4 // 1024
        if not self.device.execute_task("Random Generation", ram_needed, int(1e6), 15):
            return False
        
        self.raw_bits = np.random.randint(2, size=self.block_size)
        self.bases = np.random.randint(2, size=self.block_size)
        return True
    
    def prepare_and_send(self, receiver: 'EnhancedQKDParty', channel: ChannelMetrics, 
                        eavesdropper: Optional[EavesdropperSimulator] = None) -> bool:
        """Prepare and send quantum states"""
        ram_needed = self.block_size * 8 // 1024
        if not self.device.execute_task("State Preparation", ram_needed, int(5e6), 25):
            return False
        
        receiver.received_states = []
        
        for i, (bit, basis) in enumerate(zip(self.raw_bits, self.bases)):
            qc = QuantumCircuit(1, 1)
            
            if basis == 0:
                if bit == 1:
                    qc.x(0)
            else:
                if bit == 0:
                    qc.h(0)
                else:
                    qc.x(0)
                    qc.h(0)
            
            if eavesdropper:
                qc = eavesdropper.apply_attack(qc, basis, bit)
            
            if np.random.rand() > channel.transmittance:
                qc.reset(0)
                if np.random.rand() < 0.5:
                    qc.x(0)
            
            if np.random.rand() < channel.noise / 10000:
                qc.x(0)
            
            receiver.received_states.append(qc)
        
        return True
    
    def measure_received(self) -> bool:
        """Measure received quantum states"""
        ram_needed = len(self.received_states) * 4 // 1024
        if not self.device.execute_task("Measurement", ram_needed, int(2e6), 20):
            return False
        
        self.measured_bits = []
        self.measure_bases = np.random.randint(2, size=len(self.received_states))
        
        for qc, my_basis in zip(self.received_states, self.measure_bases):
            circ = qc.copy()
            
            if my_basis == 1:
                circ.h(0)
            
            circ.measure(0, 0)
            transpiled_circ = transpile(circ, self.backend)
            job = self.backend.run(transpiled_circ, shots=1)
            result = job.result().get_counts()
            
            bit = int(list(result.keys())[0]) if result else 0
            self.measured_bits.append(bit)
        
        return True
    
    def sift_keys(self, sender_bases: np.ndarray) -> bool:
        """Perform key sifting"""
        ram_needed = max(len(self.measured_bits), len(sender_bases)) * 4 // 1024
        if not self.device.execute_task("Key Sifting", ram_needed, int(1e5), 5):
            return False
        
        sifted_bits = []
        for i, (my_basis, sender_basis) in enumerate(zip(self.measure_bases, sender_bases)):
            if my_basis == sender_basis:
                sifted_bits.append(self.measured_bits[i])
        
        self.sifted_key = np.array(sifted_bits)
        return True
    
    def estimate_qber(self, peer_samples: np.ndarray, sample_indices: np.ndarray) -> float:
        """Estimate QBER from sample comparison"""
        my_samples = self.sifted_key[sample_indices]
        errors = np.sum(my_samples != peer_samples)
        return errors / len(peer_samples) if len(peer_samples) > 0 else 0.0
    
    def error_correction_and_amplification(self, peer_key: np.ndarray) -> bool:
        """Perform error correction and privacy amplification"""
        ram_needed = len(self.sifted_key) * 8 // 1024
        if not self.device.execute_task("Error Correction", ram_needed, int(1e7), 30):
            return False
        
        corrected_self, corrected_peer, leakage = self.error_corrector.correct_errors(
            self.sifted_key, peer_key
        )
        
        final_length = max(1, int(len(corrected_self) * (1 - leakage - 0.1)))
        
        bitstring = ''.join(str(b) for b in corrected_self)
        digest = hashlib.sha256(bitstring.encode()).hexdigest()
        bin_digest = bin(int(digest, 16))[2:].zfill(256)
        
        self.final_key = np.array([int(b) for b in bin_digest[:final_length]])
        return True
