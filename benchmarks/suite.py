import time
import numpy as np
from typing import Optional, Dict, List
import csv
import matplotlib.pyplot as plt

from src.qkd.device import ResourceLimitedDevice
from src.qkd.channel import ChannelMetrics, EavesdropperSimulator, AttackType
from src.qkd.party import EnhancedQKDParty
from src.qkd.error_correction import ErrorCorrectionType
from src.classical.ecdh import ClassicalECDHBaseline, CRYPTO_AVAILABLE

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
        
        alice = EnhancedQKDParty("Alice", alice_device, block_size, error_correction)
        bob = EnhancedQKDParty("Bob", bob_device, block_size, error_correction)
        
        try:
            if not alice.generate_random_data() or not bob.generate_random_data():
                return None
            
            if not alice.prepare_and_send(bob, channel, eavesdropper):
                return None
            
            if not bob.measure_received():
                return None
            
            if not bob.sift_keys(alice.bases):
                return None
            
            if not alice.sift_keys(alice.bases):
                alice.sifted_key = alice.raw_bits[alice.bases == bob.measure_bases[:len(alice.bases)]]
            
            if len(bob.sifted_key) == 0:
                return None
            
            sample_size = min(100, len(bob.sifted_key) // 10)
            if sample_size == 0:
                return None
            
            sample_indices = np.random.choice(len(bob.sifted_key), sample_size, replace=False)
            alice_samples = alice.sifted_key[sample_indices] if len(alice.sifted_key) > max(sample_indices) else np.random.randint(2, size=sample_size)
            qber = bob.estimate_qber(alice_samples, sample_indices)
            
            if not alice.error_correction_and_amplification(bob.sifted_key):
                return None
            
            if not bob.error_correction_and_amplification(alice.sifted_key):
                return None
            
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
                'alice_ram_usage': alice.device.allocated_ram / 1024,
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
            if not alice_ecdh.generate_keypair() or not bob_ecdh.generate_keypair():
                return None
            
            if not alice_ecdh.compute_shared_secret(bob_ecdh.public_key):
                return None
            
            if not bob_ecdh.compute_shared_secret(alice_ecdh.public_key):
                return None
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = alice_ecdh.shared_secret == bob_ecdh.shared_secret
            
            return {
                'protocol': 'ECDH',
                'key_length': 256,
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
        
        device_configs = [
            {"name": "Low-end", "cpu": 50, "ram": 32, "flash": 128, "power": 20},
            {"name": "Mid-range", "cpu": 80, "ram": 64, "flash": 256, "power": 35},
            {"name": "High-end", "cpu": 120, "ram": 128, "flash": 512, "power": 50}
        ]
        
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
            
            classical_result = self.run_classical_benchmark(alice_device, bob_device)
            if classical_result:
                classical_result.update(device_config)
                results.append(classical_result)
            
            alice_device.total_cpu_cycles = 0
            alice_device.total_power_consumed = 0
            bob_device.total_cpu_cycles = 0
            bob_device.total_power_consumed = 0
            
            for block_size in block_sizes:
                for error_correction in error_corrections:
                    for attack_type, attack_strength in attack_scenarios:
                        
                        base_transmittance = 0.85
                        base_noise = 200
                        
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
                        
                        result = self.run_qkd_benchmark(
                            alice_device, bob_device, channel, block_size, 
                            error_correction, eavesdropper
                        )
                        
                        if result:
                            result.update(device_config)
                            results.append(result)
                        
                        alice_device.total_cpu_cycles = 0
                        alice_device.total_power_consumed = 0
                        bob_device.total_cpu_cycles = 0
                        bob_device.total_power_consumed = 0
        
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

        plt.figure()
        plt.scatter([r['attack_strength'] for r in qkd_results], [r['qber'] for r in qkd_results])
        plt.xlabel("Attack Strength")
        plt.ylabel("QBER")
        plt.title("QBER vs Attack Strength")
        plt.grid(True)
        plt.savefig("qber_vs_attack_strength.png")

        plt.figure()
        plt.plot([r['block_size'] for r in qkd_results], [r['key_rate_bps'] for r in qkd_results], 'o-')
        plt.xlabel("Block Size")
        plt.ylabel("Key Rate (bps)")
        plt.title("Key Rate vs Block Size")
        plt.grid(True)
        plt.savefig("key_rate_vs_block_size.png")

        plt.figure()
        plt.scatter([r['qber'] for r in qkd_results], [r['final_key_length'] for r in qkd_results])
        plt.xlabel("QBER")
        plt.ylabel("Final Key Length")
        plt.title("Final Key Length vs QBER")
        plt.grid(True)
        plt.savefig("key_length_vs_qber.png")

        print("Plots saved as PNG files.")