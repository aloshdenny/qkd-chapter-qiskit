# 1. Initialization & Resource Assessment
# 
# a. Capability exchange (CPU, memory, power, QRNG rate)
# b. Channel probing (transmittance, noise, QBER estimation)
# c. Parameter negotiation (block size, error-correction choice, decoy states, sync tolerance)

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
import numpy as np

class IoTDevice:
    def __init__(self, name, cpu_freq_mhz, flash_kb, ram_kb, power_mw, interface, qrng_rate_mbps):
        self.name = name
        # hardware capabilities
        self.cpu_freq = cpu_freq_mhz          # MHz
        self.flash = flash_kb                # KB
        self.ram = ram_kb                    # KB
        self.power = power_mw                # mW average
        self.interface = interface           # e.g., 'LoRaWAN', '802.15.4'
        self.qrng_rate = qrng_rate_mbps      # Mbps
        # runtime parameters to be negotiated
        self.params = {}

    def compute_resource_index(self):
        # Normalize metrics and compute a simple index
        norm_cpu = self.cpu_freq / 168.0
        norm_ram = self.ram / 512.0
        norm_power = (50.0 - self.power) / 50.0
        norm_qrng = self.qrng_rate / 10.0
        # weighted sum
        return 0.3*norm_cpu + 0.3*norm_ram + 0.2*norm_power + 0.2*norm_qrng

    def send_capabilities(self):
        cap = {
            'cpu_freq': self.cpu_freq,
            'flash': self.flash,
            'ram': self.ram,
            'power': self.power,
            'interface': self.interface,
            'qrng_rate': self.qrng_rate,
            'resource_index': self.compute_resource_index()
        }
        return cap

    def receive_capabilities(self, peer_caps):
        self.peer_caps = peer_caps

    def negotiate_params(self, channel_metrics):
        # choose block size proportional to weakest resource index
        r_self = self.compute_resource_index()
        r_peer = self.peer_caps['resource_index']
        min_index = min(r_self, r_peer)
        self.params['block_size'] = int(min_index * 5000)
        # error correction code choice based on estimated QBER from channel metrics
        qber = channel_metrics['qber_est']
        if qber < 0.05:
            self.params['error_code'] = 'BCH'
        elif qber < 0.08:
            self.params['error_code'] = 'LDPC'
        else:
            self.params['error_code'] = 'RETRANSMIT'
        # decoy intensities and probabilities
        self.params['decoy_intensities'] = [0.1, 0.5, 0.9]
        self.params['decoy_probs'] = [0.1, 0.8, 0.1]
        # synchronization window (ns)
        self.params['sync_tol_ns'] = 100
        return self.params

# Simulated channel probe function
def simulate_channel_probe(num_probes=100):
    # simulate transmittance between 0.8 and 0.95
    trans = np.random.uniform(0.8, 0.95)
    # simulate background noise (counts per second)
    noise = np.random.uniform(100, 500)
    # estimate QBER based on noise
    qber_est = min(0.15, noise / 1000)
    return {'transmittance': trans, 'noise': noise, 'qber_est': qber_est}

# Example usage
if __name__ == '__main__':
    # instantiate Alice and Bob
    alice = IoTDevice('Alice', cpu_freq_mhz=120, flash_kb=1024, ram_kb=256, power_mw=30, interface='802.15.4', qrng_rate_mbps=5)
    bob   = IoTDevice('Bob',   cpu_freq_mhz=80,  flash_kb=512,  ram_kb=128, power_mw=40, interface='802.15.4', qrng_rate_mbps=3)

    # Phase 1: Initialization & Resource Assessment
    cap_a = alice.send_capabilities()
    cap_b = bob.send_capabilities()

    alice.receive_capabilities(cap_b)
    bob.receive_capabilities(cap_a)

    channel_metrics = simulate_channel_probe()

    params_a = alice.negotiate_params(channel_metrics)
    params_b = bob.negotiate_params(channel_metrics)

    print("Alice negotiated params:", params_a)
    print("Bob negotiated params:", params_b)

# 2. State Preparation, Transmission & Sifting
# 
# a. Bit & Basis Generation (Alice & Bob) via QRNG
# 
# b. State Preparation & Noisy Channel (X/Z bases, optional bit-flip noise)
# 
# c. Measurement (Bob picks random bases & measures)
# 
# d. Sifting (Alice and Bob keep only bits where their bases match)

class QKDParty:
    def __init__(self, name, block_size, qrng_rate_mbps,
                 decoy_intensities=None, decoy_probs=None):
        self.name = name
        self.block_size = block_size
        self.backend = Aer.get_backend('qasm_simulator')
        self.qrng_rate_mbps = qrng_rate_mbps
        # bits, bases, intensities
        self.raw_bits = None
        self.bases = None
        self.intensities = None
        self.measured_bits = None
        # decoy-state settings
        if decoy_intensities is None:
            self.decoy_intensities = [0.1, 0.5, 0.9]
            self.decoy_probs = [0.1, 0.8, 0.1]
        else:
            self.decoy_intensities = decoy_intensities
            self.decoy_probs = decoy_probs

    def generate_bits_bases_intensities(self):
        # simulate random bit, basis, and decoy intensity selection
        self.raw_bits = np.random.randint(2, size=self.block_size)
        self.bases = np.random.randint(2, size=self.block_size)
        self.intensities = np.random.choice(
            self.decoy_intensities,
            size=self.block_size,
            p=self.decoy_probs
        )
        return self.raw_bits, self.bases, self.intensities

    def prepare_and_send(self, receiver, channel_metrics):
        # simulate sending qubits with decoy intensities and channel properties
        receiver.received_states = []
        transmittance = channel_metrics.get('transmittance', 0.9)
        noise = channel_metrics.get('noise', 200)
        for bit, basis, intensity in zip(self.raw_bits, self.bases, self.intensities):
            qc = QuantumCircuit(1, 1)
            # state preparation
            if basis == 0:  # Z-basis
                if bit == 1:
                    qc.x(0)
            else:  # X-basis
                if bit == 0:
                    qc.h(0)
                else:
                    qc.x(0)
                    qc.h(0)
            # simulate channel loss: each qubit survives with probability transmittance * intensity
            survive_prob = transmittance * intensity
            if np.random.rand() > survive_prob:
                # photon lost: represent as random vacuum trigger -> random bit
                lost_bit = np.random.randint(2)
                qc.reset(0)
                if lost_bit == 1:
                    qc.x(0)
            # dark count / background noise induced error
            dark_prob = noise / 1e4  # scaled
            if np.random.rand() < dark_prob:
                qc.x(0)
            # store for receiver measurement
            receiver.received_states.append(qc)
        return True

    def measure_received(self):
        # receiver measures each stored circuit in random basis
        self.measured_bits = []
        self.measure_bases = np.random.randint(2, size=len(self.received_states))
        for qc, my_basis in zip(self.received_states, self.measure_bases):
            # create a copy to avoid altering original circuit
            circ = qc.copy()
            # apply measurement basis
            if my_basis == 1:
                circ.h(0)
            circ.measure(0, 0)
            transpiled_circ = transpile(circ, self.backend)
            job = self.backend.run(transpiled_circ, shots=1)
            result = job.result().get_counts()
            bit = int(list(result.keys())[0])
            self.measured_bits.append(bit)
        return np.array(self.measured_bits), self.measure_bases
    
    def sift_raw(self, sender_bases):
        """Sift raw_bits against sender_bases (no measured data involved)."""
        sifted_key = []
        sifted_indices = []
        for i, my_basis in enumerate(self.bases):
            if my_basis == sender_bases[i]:
                sifted_key.append(self.raw_bits[i])
                sifted_indices.append(i)
        return np.array(sifted_key), sifted_indices

    def sift_keys(self, sender_bases, sender_intensities=None, reveal_decoys=False):
        # compare bases, optionally reveal decoy positions
        sifted_key = []
        sifted_indices = []
        for i, my_basis in enumerate(self.measure_bases):
            if my_basis == sender_bases[i]:
                # if decoy-state: optionally discard decoy positions when generating final key
                if reveal_decoys and sender_intensities is not None:
                    # reveal intensities and filter only signal states (highest intensity)
                    if sender_intensities[i] == max(self.decoy_intensities):
                        sifted_key.append(self.measured_bits[i])
                        sifted_indices.append(i)
                else:
                    sifted_key.append(self.measured_bits[i])
                    sifted_indices.append(i)
        return np.array(sifted_key), sifted_indices

# Example flow for Phase 2 with decoy-state and enhanced noise
if __name__ == '__main__':
    block_size = 1000
    channel_metrics = {'transmittance': 0.85, 'noise': 300}

    alice = QKDParty('Alice', block_size, qrng_rate_mbps=5,
                     decoy_intensities=[0.1, 0.5, 0.9],
                     decoy_probs=[0.1, 0.8, 0.1])
    bob = QKDParty('Bob', block_size, qrng_rate_mbps=3,
                   decoy_intensities=[0.1, 0.5, 0.9],
                   decoy_probs=[0.1, 0.8, 0.1])

    # 2a: generate bits, bases, intensities
    bits_a, bases_a, ints_a = alice.generate_bits_bases_intensities()

    # 2b: prepare and send
    alice.prepare_and_send(bob, channel_metrics)

    # 2c: Bob measures
    bits_b, bases_b = bob.measure_received()

    # 2d: sifting without decoy filtering
    sifted_a, idx = bob.sift_keys(bases_a)
    sifted_b, _ = bob.sift_keys(bases_a)
    print(f"Raw sifted length: {len(sifted_a)}")

    # 2e: decoy-state analysis (reveal decoys and keep only signal states)
    key_a, indices = bob.sift_keys(bases_a, ints_a, reveal_decoys=True)
    key_b, _ = bob.sift_keys(bases_a, ints_a, reveal_decoys=True)
    print(f"Signal-key length: {len(key_a)}")

    # mismatch check
    mismatches = np.sum(key_a != key_b)
    print(f"Signal key mismatches: {mismatches}")

# 3. Error Estimation, Reconciliation & Privacy Amplification
# 
# a. Sampling & QBER Estimation (3a)
# 
# b. Error Correction via simple parity-based syndrome exchange (3b)
# 
# c. Privacy Amplification using SHA-256 hashing and truncation (3c)

import numpy as np
import hashlib

class ReconciliationParty:
    def __init__(self, name, sifted_key, error_code='BCH', block_size=None):
        self.name = name
        self.sifted_key = np.array(sifted_key, dtype=int)
        self.error_code = error_code
        self.block_size = block_size or len(self.sifted_key)

    def sample_for_qber(self, sample_fraction=0.1):
        # randomly select positions to disclose and estimate QBER
        num_samples = int(len(self.sifted_key) * sample_fraction)
        indices = np.random.choice(len(self.sifted_key), num_samples, replace=False)
        sample_bits = self.sifted_key[indices]
        return indices, sample_bits

    def estimate_qber(self, alice_samples, bob_samples):
        # assume alice_samples and bob_samples are aligned arrays
        mismatches = np.sum(alice_samples != bob_samples)
        qber = mismatches / len(alice_samples)
        return qber

    def error_correction(self, peer_key, syndrome_exchange=True):
        # For simulation, we simply reconcile by aligning keys and correcting single-bit errors
        corrected = self.sifted_key.copy()
        # simple parity-based correction for demonstration
        if syndrome_exchange:
            # compute parity
            p_self = np.sum(corrected) % 2
            p_peer = np.sum(peer_key) % 2
            if p_self != p_peer and len(corrected) > 0:
                # flip a random bit to mimic correction
                flip_idx = np.random.randint(len(corrected))
                corrected[flip_idx] ^= 1
        return corrected

    def estimate_leakage(self):
        # base leakage rates per code, plus random fluctuation
        base = {'BCH': 0.1, 'LDPC': 0.2}.get(self.error_code, 0.3)
        # add small Gaussian noise to simulate randomness
        leakage = base + np.random.normal(0, 0.02)
        # clip to [0,1]
        return float(np.clip(leakage, 0.0, 1.0))

    def privacy_amplification(self, corrected_key, final_key_length=None):
        # Use a universal hash (SHA256) and truncate
        bitstring = ''.join(str(b) for b in corrected_key)
        digest = hashlib.sha256(bitstring.encode()).hexdigest()
        # convert hex digest to binary string
        bin_digest = bin(int(digest, 16))[2:].zfill(256)
        # determine final length based on leakage estimate
        if final_key_length is None:
            leak = self.estimate_leakage()
            final_key_length = int(len(corrected_key) * (1 - leak))
        return np.array(list(map(int, bin_digest[:final_key_length])))

# Example flow for Phase 3 with random leakage model
def main():
    # dummy sifted keys from Alice and Bob
    alice_sifted = np.random.randint(2, size=900)
    bob_sifted = alice_sifted.copy()
    # introduce some errors
    error_positions = np.random.choice(900, 10, replace=False)
    bob_sifted[error_positions] ^= 1

    alice = ReconciliationParty('Alice', alice_sifted, error_code='LDPC')
    bob = ReconciliationParty('Bob', bob_sifted, error_code='LDPC')

    # 3a: sampling for QBER
    idx_a, samples_a = alice.sample_for_qber(0.1)
    idx_b, samples_b = bob.sample_for_qber(0.1)
    # align samples by indices (for simplicity assume same indices selected)
    qber = alice.estimate_qber(samples_a, samples_b)
    print(f"Estimated QBER: {qber:.3f}")

    # 3b: error correction
    alice_corrected = alice.error_correction(bob.sifted_key)
    bob_corrected = bob.error_correction(alice.sifted_key)
    mismatches_after = np.sum(alice_corrected != bob_corrected)
    print(f"Mismatches after error correction: {mismatches_after}")

    # 3c: privacy amplification
    final_alice = alice.privacy_amplification(alice_corrected)
    final_bob = bob.privacy_amplification(bob_corrected)
    print(f"Final key length (Alice): {len(final_alice)}")
    print(f"Final key length (Bob):   {len(final_bob)}")
    print(f"Key match? {np.array_equal(final_alice, final_bob)}")

if __name__ == '__main__':
    main()

# Key Confirmation & Authentication
# 
# a. Key Hashing: SHA-256 of the final bitstring
# 
# b. Authentication Tags: HMAC-SHA256 over each hash using a pre-shared auth_key
# 
# c. Verification: Constant-time tag checks and hash comparison
# 
# d. Final Acceptance/Abort based on matching hashes and valid tags

import numpy as np
import hashlib
import hmac

class AuthenticationParty:
    def __init__(self, name, final_key, auth_key):
        self.name = name
        self.final_key = np.array(final_key, dtype=int)
        self.auth_key = auth_key  # pre-shared symmetric authentication key (bytes)

    def compute_key_hash(self):
        # compute SHA256 hash of final key bits
        bitstring = ''.join(str(b) for b in self.final_key)
        digest = hashlib.sha256(bitstring.encode()).digest()
        return digest

    def generate_auth_tag(self, message):
        # HMAC-SHA256 over message using auth_key
        tag = hmac.new(self.auth_key, message, hashlib.sha256).digest()
        return tag

    def verify_auth_tag(self, message, tag):
        # constant-time comparison
        expected = hmac.new(self.auth_key, message, hashlib.sha256).digest()
        return hmac.compare_digest(expected, tag)

# Example flow for Phase 4: Key Confirmation & Authentication
def main():
    # simulate final keys from Phase 3
    final_alice = np.random.randint(2, size=256)
    final_bob = final_alice.copy()
    # pre-shared authentication key (for demonstration)
    auth_key = b'supersecretsharedkey'

    alice = AuthenticationParty('Alice', final_alice, auth_key)
    bob   = AuthenticationParty('Bob', final_bob, auth_key)

    # 4a: Each party computes hash of their final key
    hash_alice = alice.compute_key_hash()
    hash_bob   = bob.compute_key_hash()
    print(f"Alice key hash: {hash_alice.hex()}")
    print(f"Bob   key hash: {hash_bob.hex()}")

    # 4b: Exchange hashes over authenticated channel (simulate by generating auth tags)
    tag_alice_hash = alice.generate_auth_tag(hash_alice)
    tag_bob_hash   = bob.generate_auth_tag(hash_bob)

    # 4c: Verify received hashes
    ok_alice = alice.verify_auth_tag(hash_bob, tag_bob_hash)
    ok_bob   = bob.verify_auth_tag(hash_alice, tag_alice_hash)
    print(f"Alice verifies Bob's hash: {ok_alice}")
    print(f"Bob verifies Alice's hash: {ok_bob}")

    # 4d: Final key acceptance
    if ok_alice and ok_bob and hash_alice == hash_bob:
        print("Key confirmed and authenticated. Secure key established.")
    else:
        print("Key confirmation failed! Abort protocol.")

if __name__ == '__main__':
    main()


import time
import numpy as np
import sys

class ResourceLimitedDevice:
    def __init__(self, name, cpu_freq_mhz, ram_kb, flash_kb, power_mw):
        self.name = name
        self.cpu_freq = cpu_freq_mhz      # in MHz
        self.ram_limit = ram_kb * 1024   # in bytes
        self.flash_limit = flash_kb * 1024
        self.power_limit = power_mw       # mW (simulated)
        self.allocated_ram = 0
        self.tasks = []

    def allocate_ram(self, size_bytes):
        """Simulate RAM allocation; raise MemoryError if exceeded."""
        if self.allocated_ram + size_bytes > self.ram_limit:
            raise MemoryError(f"{self.name}: RAM limit exceeded ({self.allocated_ram + size_bytes}/{self.ram_limit} bytes)")
        self.allocated_ram += size_bytes
        return bytearray(size_bytes)

    def free_ram(self, size_bytes):
        self.allocated_ram = max(0, self.allocated_ram - size_bytes)

    def cpu_delay(self, cycles):
        """Simulate computation time based on CPU frequency."""
        # cycles = number of CPU cycles; 1 MHz = 1e6 cycles per second
        seconds = cycles / (self.cpu_freq * 1e6)
        time.sleep(seconds)

    def simulate_task(self, name, ram_usage_kb, cpu_cycles, power_use_mw, func, *args, **kwargs):
        """Run a task with resource checks and simulate delays."""
        # RAM
        ram_bytes = ram_usage_kb * 1024
        buffer = None
        try:
            buffer = self.allocate_ram(ram_bytes)
        except MemoryError as e:
            print(e)
            return None
        # CPU
        print(f"{self.name}: Starting '{name}' (RAM {ram_usage_kb}KB, {cpu_cycles} cycles)")
        self.cpu_delay(cpu_cycles)
        # Power check (simulate simple threshold)
        if power_use_mw > self.power_limit:
            print(f"{self.name}: Power limit exceeded for task '{name}' ({power_use_mw}mW > {self.power_limit}mW)")
            result = None
        else:
            # execute actual function
            result = func(*args, **kwargs)
        # cleanup
        self.free_ram(ram_bytes)
        print(f"{self.name}: Finished '{name}'")
        return result

# Example integration with QKD phases
if __name__ == '__main__':
    # wrap IoTDevice inside ResourceLimitedDevice
    alice_hw = ResourceLimitedDevice('AliceHW', cpu_freq_mhz=80, ram_kb=128, flash_kb=512, power_mw=30)
    bob_hw = ResourceLimitedDevice('BobHW', cpu_freq_mhz=60, ram_kb=64, flash_kb=256, power_mw=40)

    def run_phase1():
        # instantiate logical devices
        alice = IoTDevice('Alice', cpu_freq_mhz=120, flash_kb=1024, ram_kb=256, power_mw=30, interface='802.15.4', qrng_rate_mbps=5)
        bob   = IoTDevice('Bob',   cpu_freq_mhz=80,  flash_kb=512,  ram_kb=128, power_mw=40, interface='802.15.4', qrng_rate_mbps=3)
        cap_a = alice.send_capabilities()
        cap_b = bob.send_capabilities()
        alice.receive_capabilities(cap_b)
        bob.receive_capabilities(cap_a)
        metrics = simulate_channel_probe()
        params_a = alice.negotiate_params(metrics)
        params_b = bob.negotiate_params(metrics)
        return params_a, params_b

    # simulate phase1 under hardware constraints
    params = alice_hw.simulate_task(
        name='Phase1 Negotiation',
        ram_usage_kb=50,
        cpu_cycles=5e7,
        power_use_mw=25,
        func=run_phase1
    )
    print('Negotiated params:', params)