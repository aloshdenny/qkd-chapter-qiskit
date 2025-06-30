# Enhanced ARC QKD Protocol with Eavesdropper Detection and IoT Analysis
# 
# Improvements:
# 1. Realistic eavesdropping attack simulation
# 2. Enhanced error correction with multiple algorithms
# 3. Automated benchmark suite for performance analysis
# 4. Classical ECDH baseline comparison
# 5. Comprehensive feasibility study framework

import numpy as np
import hashlib
import hmac
import time
import csv
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum
import json

# Qiskit imports for quantum simulation
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

# Classical crypto for comparison
try:
    from Crypto.PublicKey import ECC
    from Crypto.Hash import SHA256
    from Crypto.Protocol.KDF import HKDF
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: PyCryptodome not available, classical comparison disabled")

class AttackType(Enum):
    NONE = "none"
    INTERCEPT_RESEND = "intercept_resend"
    PHOTON_SPLITTING = "photon_splitting"
    BEAM_SPLITTING = "beam_splitting"

class ErrorCorrectionType(Enum):
    PARITY = "parity"
    BCH = "bch"
    LDPC = "ldpc"
    CASCADE = "cascade"

@dataclass
class ChannelMetrics:
    transmittance: float
    noise: float
    qber_est: float
    attack_type: AttackType = AttackType.NONE
    attack_strength: float = 0.0

@dataclass
class PerformanceMetrics:
    key_rate_bps: float
    qber: float
    ram_usage_kb: float
    cpu_cycles: int
    power_mw: float
    success_rate: float
    final_key_length: int
    raw_key_length: int

class EavesdropperSimulator:
    """Simulates various eavesdropping attacks on the quantum channel"""
    
    def __init__(self, attack_type: AttackType, strength: float = 0.5):
        self.attack_type = attack_type
        self.strength = strength  # 0.0 = no attack, 1.0 = maximum attack
    
    def apply_attack(self, qc: QuantumCircuit, basis: int, bit: int) -> QuantumCircuit:
        """Apply eavesdropping attack to quantum circuit"""
        attacked_qc = qc.copy()
        
        if self.attack_type == AttackType.INTERCEPT_RESEND:
            # Eve measures in random basis and resends
            if np.random.rand() < self.strength:
                eve_basis = np.random.randint(2)
                # Measure in Eve's basis
                if eve_basis == 1:  # X-basis
                    attacked_qc.h(0)
                attacked_qc.measure_all()
                
                # Resend based on measurement (simplified)
                eve_bit = np.random.randint(2)  # Simulate measurement result
                attacked_qc = QuantumCircuit(1, 1)
                if eve_basis == 0:  # Z-basis
                    if eve_bit == 1:
                        attacked_qc.x(0)
                else:  # X-basis
                    if eve_bit == 0:
                        attacked_qc.h(0)
                    else:
                        attacked_qc.x(0)
                        attacked_qc.h(0)
        
        elif self.attack_type == AttackType.PHOTON_SPLITTING:
            # Simulate photon number splitting attack
            if np.random.rand() < self.strength * 0.3:  # Lower probability but higher impact
                # Introduce phase error
                attacked_qc.rz(np.pi * np.random.rand(), 0)
        
        elif self.attack_type == AttackType.BEAM_SPLITTING:
            # Simulate beam splitter attack
            if np.random.rand() < self.strength * 0.4:
                # Add amplitude damping
                attacked_qc.reset(0)
                if np.random.rand() < 0.5:
                    attacked_qc.x(0)
        
        return attacked_qc

class EnhancedErrorCorrection:
    """Enhanced error correction with multiple algorithms"""
    
    def __init__(self, method: ErrorCorrectionType):
        self.method = method
    
    def correct_errors(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Perform error correction and return corrected keys and leakage"""
        
        if self.method == ErrorCorrectionType.PARITY:
            return self._parity_correction(alice_key, bob_key)
        elif self.method == ErrorCorrectionType.BCH:
            return self._bch_correction(alice_key, bob_key)
        elif self.method == ErrorCorrectionType.LDPC:
            return self._ldpc_correction(alice_key, bob_key)
        elif self.method == ErrorCorrectionType.CASCADE:
            return self._cascade_correction(alice_key, bob_key)
        else:
            return alice_key, bob_key, 0.0
    
    def _parity_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simple parity-based correction"""
        corrected_alice = alice_key.copy()
        corrected_bob = bob_key.copy()
        
        # Check overall parity
        p_alice = np.sum(corrected_alice) % 2
        p_bob = np.sum(corrected_bob) % 2
        
        if p_alice != p_bob and len(corrected_bob) > 0:
            # Flip random bit in Bob's key
            flip_idx = np.random.randint(len(corrected_bob))
            corrected_bob[flip_idx] ^= 1
        
        leakage = 1.0 / len(alice_key) if len(alice_key) > 0 else 0.0
        return corrected_alice, corrected_bob, leakage
    
    def _bch_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate BCH code correction"""
        # Simplified BCH simulation - in practice would use proper BCH implementation
        error_positions = np.where(alice_key != bob_key)[0]
        corrected_bob = bob_key.copy()
        
        # BCH can correct up to t errors where 2t+1 <= d (minimum distance)
        # Assume BCH(n,k,t) with t=3 capability
        max_correctable = 3
        
        if len(error_positions) <= max_correctable:
            corrected_bob[error_positions] = alice_key[error_positions]
        
        # BCH leakage approximately log2(n choose t) bits per block
        n = len(alice_key)
        leakage = min(0.15, max_correctable * np.log2(n) / n) if n > 0 else 0.0
        
        return alice_key, corrected_bob, leakage
    
    def _ldpc_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate LDPC correction"""
        # Simplified LDPC simulation
        error_positions = np.where(alice_key != bob_key)[0]
        corrected_bob = bob_key.copy()
        
        # LDPC iterative correction (simplified)
        max_iterations = 10
        for _ in range(max_iterations):
            if len(error_positions) == 0:
                break
            # Correct some errors in each iteration
            correct_count = min(len(error_positions), max(1, len(error_positions) // 3))
            correct_indices = np.random.choice(error_positions, correct_count, replace=False)
            corrected_bob[correct_indices] = alice_key[correct_indices]
            error_positions = np.where(alice_key != corrected_bob)[0]
        
        # LDPC leakage rate
        leakage = 0.12 + np.random.normal(0, 0.02)
        leakage = float(np.clip(leakage, 0.0, 0.3))
        
        return alice_key, corrected_bob, leakage
    
    def _cascade_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate Cascade protocol"""
        corrected_alice = alice_key.copy()
        corrected_bob = bob_key.copy()
        
        # Cascade protocol simulation
        block_sizes = [len(alice_key) // 4, len(alice_key) // 8, len(alice_key) // 16]
        total_leaked = 0
        
        for block_size in block_sizes:
            if block_size < 2:
                continue
                
            num_blocks = len(corrected_alice) // block_size
            for i in range(num_blocks):
                start = i * block_size
                end = start + block_size
                
                alice_block = corrected_alice[start:end]
                bob_block = corrected_bob[start:end]
                
                # Check parity
                p_alice = np.sum(alice_block) % 2
                p_bob = np.sum(bob_block) % 2
                total_leaked += 1  # Parity bit revealed
                
                if p_alice != p_bob:
                    # Binary search for error (simplified)
                    error_pos = np.random.randint(len(bob_block))
                    corrected_bob[start + error_pos] ^= 1
                    total_leaked += int(np.log2(block_size))
        
        leakage = total_leaked / len(alice_key) if len(alice_key) > 0 else 0.0
        return corrected_alice, corrected_bob, leakage

class ResourceLimitedDevice:
    """Enhanced resource-constrained device simulation"""
    
    def __init__(self, name: str, cpu_freq_mhz: int, ram_kb: int, flash_kb: int, power_mw: int):
        self.name = name
        self.cpu_freq = cpu_freq_mhz
        self.ram_limit = ram_kb * 1024
        self.flash_limit = flash_kb * 1024
        self.power_limit = power_mw
        self.allocated_ram = 0
        self.total_cpu_cycles = 0
        self.total_power_consumed = 0.0
    
    def allocate_ram(self, size_bytes: int) -> bool:
        """Try to allocate RAM, return success status"""
        if self.allocated_ram + size_bytes > self.ram_limit:
            return False
        self.allocated_ram += size_bytes
        return True
    
    def free_ram(self, size_bytes: int):
        """Free allocated RAM"""
        self.allocated_ram = max(0, self.allocated_ram - size_bytes)
    
    def execute_task(self, name: str, ram_kb: int, cpu_cycles: int, power_mw: int) -> bool:
        """Execute a task with resource constraints"""
        ram_bytes = ram_kb * 1024
        
        # Check RAM availability
        if not self.allocate_ram(ram_bytes):
            print(f"{self.name}: Insufficient RAM for {name} ({ram_kb}KB needed, {(self.ram_limit - self.allocated_ram)//1024}KB available)")
            return False
        
        # Check power constraint
        if power_mw > self.power_limit:
            print(f"{self.name}: Power limit exceeded for {name} ({power_mw}mW > {self.power_limit}mW)")
            self.free_ram(ram_bytes)
            return False
        
        # Simulate execution time
        execution_time = cpu_cycles / (self.cpu_freq * 1e6)
        time.sleep(min(execution_time, 0.01))  # Cap simulation delay
        
        # Update counters
        self.total_cpu_cycles += cpu_cycles
        self.total_power_consumed += power_mw * execution_time / 1000  # mJ
        
        # Free RAM
        self.free_ram(ram_bytes)
        return True

class EnhancedQKDParty:
    """Enhanced QKD party with attack detection and multiple error correction"""
    
    def __init__(self, name: str, device: ResourceLimitedDevice, block_size: int, 
                 error_correction: ErrorCorrectionType = ErrorCorrectionType.LDPC):
        self.name = name
        self.device = device
        self.block_size = block_size
        self.error_corrector = EnhancedErrorCorrection(error_correction)
        self.backend = Aer.get_backend('qasm_simulator')
        
        # QKD data
        self.raw_bits = None
        self.bases = None
        self.measured_bits = None
        self.sifted_key = None
        self.final_key = None
        
        # Performance tracking
        self.metrics = PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def generate_random_data(self) -> bool:
        """Generate random bits and bases"""
        ram_needed = self.block_size * 3 * 4 // 1024  # Rough estimate for 3 arrays
        if not self.device.execute_task("Random Generation", ram_needed, int(1e6), 15):
            return False
        
        self.raw_bits = np.random.randint(2, size=self.block_size)
        self.bases = np.random.randint(2, size=self.block_size)
        return True
    
    def prepare_and_send(self, receiver: 'EnhancedQKDParty', channel: ChannelMetrics, 
                        eavesdropper: Optional[EavesdropperSimulator] = None) -> bool:
        """Prepare and send quantum states"""
        ram_needed = self.block_size * 8 // 1024  # Estimate for quantum circuits
        if not self.device.execute_task("State Preparation", ram_needed, int(5e6), 25):
            return False
        
        receiver.received_states = []
        
        for i, (bit, basis) in enumerate(zip(self.raw_bits, self.bases)):
            # Create quantum circuit
            qc = QuantumCircuit(1, 1)
            
            # State preparation
            if basis == 0:  # Z-basis
                if bit == 1:
                    qc.x(0)
            else:  # X-basis
                if bit == 0:
                    qc.h(0)
                else:
                    qc.x(0)
                    qc.h(0)
            
            # Apply eavesdropping attack if present
            if eavesdropper:
                qc = eavesdropper.apply_attack(qc, basis, bit)
            
            # Simulate channel loss and noise
            if np.random.rand() > channel.transmittance:
                # Photon lost
                qc.reset(0)
                if np.random.rand() < 0.5:
                    qc.x(0)
            
            # Dark counts
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
            
            # Apply measurement basis
            if my_basis == 1:  # X-basis
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
        
        # Error correction
        corrected_self, corrected_peer, leakage = self.error_corrector.correct_errors(
            self.sifted_key, peer_key
        )
        
        # Privacy amplification
        final_length = max(1, int(len(corrected_self) * (1 - leakage - 0.1)))  # Security margin
        
        bitstring = ''.join(str(b) for b in corrected_self)
        digest = hashlib.sha256(bitstring.encode()).hexdigest()
        bin_digest = bin(int(digest, 16))[2:].zfill(256)
        
        self.final_key = np.array([int(b) for b in bin_digest[:final_length]])
        return True

class ClassicalECDHBaseline:
    """Classical ECDH baseline for comparison"""
    
    def __init__(self, device: ResourceLimitedDevice):
        self.device = device
        self.private_key = None
        self.public_key = None
        self.shared_secret = None
    
    def generate_keypair(self) -> bool:
        """Generate ECDH key pair"""
        if not CRYPTO_AVAILABLE:
            return False
        
        if not self.device.execute_task("ECDH KeyGen", 4, int(5e7), 50):  # More CPU intensive
            return False
        
        self.private_key = ECC.generate(curve='P-256')
        self.public_key = self.private_key.public_key()
        return True
    
    def compute_shared_secret(self, peer_public_key) -> bool:
        """Compute shared secret"""
        if not CRYPTO_AVAILABLE or not self.private_key:
            return False
        
        if not self.device.execute_task("ECDH Agreement", 2, int(3e7), 40):
            return False
        
        # Simulate ECDH computation
        point = peer_public_key.pointQ * self.private_key.d
        self.shared_secret = point.x.to_bytes(32, 'big')
        return True

class QKDBenchmarkSuite:
    """Comprehensive benchmark suite for QKD analysis"""
    
    def __init__(self):
        self.results = []
    
    def run_qkd_benchmark(self, alice_device: ResourceLimitedDevice, bob_device: ResourceLimitedDevice,
                         channel: ChannelMetrics, block_size: int, 
                         error_correction: ErrorCorrectionType,
                         eavesdropper: Optional[EavesdropperSimulator] = None) -> Optional[Dict]:
        """Run a single QKD benchmark"""
        
        start_time = time.time()
        
        # Initialize parties
        alice = EnhancedQKDParty("Alice", alice_device, block_size, error_correction)
        bob = EnhancedQKDParty("Bob", bob_device, block_size, error_correction)
        
        try:
            # Phase 1: Random generation
            if not alice.generate_random_data() or not bob.generate_random_data():
                return None
            
            # Phase 2: Transmission and measurement
            if not alice.prepare_and_send(bob, channel, eavesdropper):
                return None
            
            if not bob.measure_received():
                return None
            
            # Phase 3: Key sifting
            if not bob.sift_keys(alice.bases):
                return None
            
            if not alice.sift_keys(alice.bases):  # Alice sifts her own key
                alice.sifted_key = alice.raw_bits[alice.bases == bob.measure_bases[:len(alice.bases)]]
            
            if len(bob.sifted_key) == 0:
                return None
            
            # Phase 4: QBER estimation
            sample_size = min(100, len(bob.sifted_key) // 10)
            if sample_size == 0:
                return None
            
            sample_indices = np.random.choice(len(bob.sifted_key), sample_size, replace=False)
            alice_samples = alice.sifted_key[sample_indices] if len(alice.sifted_key) > max(sample_indices) else np.random.randint(2, size=sample_size)
            qber = bob.estimate_qber(alice_samples, sample_indices)
            
            # Phase 5: Error correction and privacy amplification
            if not alice.error_correction_and_amplification(bob.sifted_key):
                return None
            
            if not bob.error_correction_and_amplification(alice.sifted_key):
                return None
            
            # Calculate metrics
            end_time = time.time()
            duration = end_time - start_time
            
            key_rate = len(alice.final_key) / duration if duration > 0 else 0
            success = np.array_equal(alice.final_key, bob.final_key) if alice.final_key is not None and bob.final_key is not None else False
            
            return {
                'block_size': block_size,
                'error_correction': error_correction.value,
                'attack_type': eavesdropper.attack_type.value if eavesdropper else 'none',
                'attack_strength': eavesdropper.strength if eavesdropper else 0.0,
                'qber': qber,
                'key_rate_bps': key_rate,
                'final_key_length': len(alice.final_key) if alice.final_key is not None else 0,
                'raw_key_length': len(alice.sifted_key) if alice.sifted_key is not None else 0,
                'success': success,
                'duration': duration,
                'alice_ram_usage': alice.device.allocated_ram / 1024,  # KB
                'bob_ram_usage': bob.device.allocated_ram / 1024,
                'alice_cpu_cycles': alice.device.total_cpu_cycles,
                'bob_cpu_cycles': bob.device.total_cpu_cycles,
                'channel_transmittance': channel.transmittance,
                'channel_noise': channel.noise
            }
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            return None
    
    def run_classical_benchmark(self, alice_device: ResourceLimitedDevice, 
                               bob_device: ResourceLimitedDevice) -> Optional[Dict]:
        """Run classical ECDH benchmark"""
        if not CRYPTO_AVAILABLE:
            return None
        
        start_time = time.time()
        
        alice_ecdh = ClassicalECDHBaseline(alice_device)
        bob_ecdh = ClassicalECDHBaseline(bob_device)
        
        try:
            # Key generation
            if not alice_ecdh.generate_keypair() or not bob_ecdh.generate_keypair():
                return None
            
            # Key agreement
            if not alice_ecdh.compute_shared_secret(bob_ecdh.public_key):
                return None
            
            if not bob_ecdh.compute_shared_secret(alice_ecdh.public_key):
                return None
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = alice_ecdh.shared_secret == bob_ecdh.shared_secret
            
            return {
                'protocol': 'ECDH',
                'key_length': 256,  # bits
                'success': success,
                'duration': duration,
                'alice_ram_usage': alice_device.allocated_ram / 1024,
                'bob_ram_usage': bob_device.allocated_ram / 1024,
                'alice_cpu_cycles': alice_device.total_cpu_cycles,
                'bob_cpu_cycles': bob_device.total_cpu_cycles
            }
            
        except Exception as e:
            print(f"Classical benchmark failed: {e}")
            return None
    
    def run_comprehensive_study(self, output_file: str = "qkd_benchmark_results.csv"):
        """Run comprehensive feasibility study"""
        
        # Device configurations
        device_configs = [
            {"name": "Low-end", "cpu": 50, "ram": 32, "flash": 128, "power": 20},
            {"name": "Mid-range", "cpu": 80, "ram": 64, "flash": 256, "power": 35},
            {"name": "High-end", "cpu": 120, "ram": 128, "flash": 512, "power": 50}
        ]
        
        # Test parameters
        block_sizes = [256, 512, 1024, 2048]
        error_corrections = [ErrorCorrectionType.PARITY, ErrorCorrectionType.BCH, ErrorCorrectionType.LDPC]
        attack_scenarios = [
            (AttackType.NONE, 0.0),
            (AttackType.INTERCEPT_RESEND, 0.3),
            (AttackType.INTERCEPT_RESEND, 0.7),
            (AttackType.PHOTON_SPLITTING, 0.5)
        ]
        
        results = []
        
        print("Running comprehensive QKD feasibility study...")
        
        for device_config in device_configs:
            print(f"\nTesting {device_config['name']} devices...")
            
            alice_device = ResourceLimitedDevice(
                f"Alice-{device_config['name']}", 
                device_config['cpu'], device_config['ram'], 
                device_config['flash'], device_config['power']
            )
            bob_device = ResourceLimitedDevice(
                f"Bob-{device_config['name']}", 
                device_config['cpu'], device_config['ram'], 
                device_config['flash'], device_config['power']
            )
            
            # Test classical baseline
            classical_result = self.run_classical_benchmark(alice_device, bob_device)
            if classical_result:
                classical_result.update(device_config)
                results.append(classical_result)
            
            # Reset device counters
            alice_device.total_cpu_cycles = 0
            alice_device.total_power_consumed = 0
            bob_device.total_cpu_cycles = 0
            bob_device.total_power_consumed = 0
            
            for block_size in block_sizes:
                for error_correction in error_corrections:
                    for attack_type, attack_strength in attack_scenarios:
                        
                        # Channel configuration
                        base_transmittance = 0.85
                        base_noise = 200
                        
                        # Adjust channel based on attack
                        if attack_type != AttackType.NONE:
                            base_transmittance *= (1 - attack_strength * 0.1)
                            base_noise *= (1 + attack_strength * 0.5)
                        
                        channel = ChannelMetrics(
                            transmittance=base_transmittance,
                            noise=base_noise,
                            qber_est=base_noise / 5000,
                            attack_type=attack_type,
                            attack_strength=attack_strength
                        )
                        
                        eavesdropper = EavesdropperSimulator(attack_type, attack_strength) if attack_type != AttackType.NONE else None
                        
                        # Run QKD benchmark
                        result = self.run_qkd_benchmark(
                            alice_device, bob_device, channel, block_size, 
                            error_correction, eavesdropper
                        )
                        
                        if result:
                            result.update(device_config)
                            results.append(result)
                        
                        # Reset device counters for next test
                        alice_device.total_cpu_cycles = 0
                        alice_device.total_power_consumed = 0
                        bob_device.total_cpu_cycles = 0
                        bob_device.total_power_consumed = 0
        
        # Save results
        if results:
            self.save_results(results, output_file)
            self.generate_analysis_plots(results)
        
        return results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save benchmark results to CSV"""
        if not results:
            return
        
        fieldnames = sorted(set().union(*(r.keys() for r in results)))

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def generate_analysis_plots(self, results: List[Dict]):
        """Generate basic analysis plots from results"""
        qkd_results = [r for r in results if r.get("protocol", "QKD") != "ECDH"]
        
        if not qkd_results:
            print("No QKD results to plot.")
            return

        # Plot: QBER vs Attack Strength
        attack_strengths = [r['attack_strength'] for r in qkd_results]
        qbers = [r['qber'] for r in qkd_results]
        plt.figure()
        plt.scatter(attack_strengths, qbers)
        plt.xlabel("Attack Strength")
        plt.ylabel("QBER")
        plt.title("QBER vs Attack Strength")
        plt.grid(True)
        plt.savefig("qber_vs_attack_strength.png")

        # Plot: Key Rate vs Block Size
        block_sizes = [r['block_size'] for r in qkd_results]
        key_rates = [r['key_rate_bps'] for r in qkd_results]
        plt.figure()
        plt.plot(block_sizes, key_rates, 'o-')
        plt.xlabel("Block Size")
        plt.ylabel("Key Rate (bps)")
        plt.title("Key Rate vs Block Size")
        plt.grid(True)
        plt.savefig("key_rate_vs_block_size.png")

        # Plot: Final Key Length vs QBER
        final_key_lengths = [r['final_key_length'] for r in qkd_results]
        plt.figure()
        plt.scatter(qbers, final_key_lengths)
        plt.xlabel("QBER")
        plt.ylabel("Final Key Length")
        plt.title("Final Key Length vs QBER")
        plt.grid(True)
        plt.savefig("key_length_vs_qber.png")

        print("Plots saved as PNG files.")

if __name__ == "__main__":
    suite = QKDBenchmarkSuite()
    results = suite.run_comprehensive_study("qkd_benchmark_results.csv")

    print("\nBenchmarking Complete.")
    print(f"{len(results)} entries saved to 'qkd_benchmark_results.csv'.")