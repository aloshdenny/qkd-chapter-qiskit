import numpy as np
from enum import Enum
from typing import Tuple

class ErrorCorrectionType(Enum):
    PARITY = "parity"
    BCH = "bch"
    LDPC = "ldpc"
    CASCADE = "cascade"

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
        
        p_alice = np.sum(corrected_alice) % 2
        p_bob = np.sum(corrected_bob) % 2
        
        if p_alice != p_bob and len(corrected_bob) > 0:
            flip_idx = np.random.randint(len(corrected_bob))
            corrected_bob[flip_idx] ^= 1
        
        leakage = 1.0 / len(alice_key) if len(alice_key) > 0 else 0.0
        return corrected_alice, corrected_bob, leakage
    
    def _bch_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate BCH code correction"""
        error_positions = np.where(alice_key != bob_key)[0]
        corrected_bob = bob_key.copy()
        
        max_correctable = 3
        
        if len(error_positions) <= max_correctable:
            corrected_bob[error_positions] = alice_key[error_positions]
        
        n = len(alice_key)
        leakage = min(0.15, max_correctable * np.log2(n) / n) if n > 0 else 0.0
        
        return alice_key, corrected_bob, leakage
    
    def _ldpc_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate LDPC correction"""
        error_positions = np.where(alice_key != bob_key)[0]
        corrected_bob = bob_key.copy()
        
        max_iterations = 10
        for _ in range(max_iterations):
            if len(error_positions) == 0:
                break
            correct_count = min(len(error_positions), max(1, len(error_positions) // 3))
            correct_indices = np.random.choice(error_positions, correct_count, replace=False)
            corrected_bob[correct_indices] = alice_key[correct_indices]
            error_positions = np.where(alice_key != corrected_bob)[0]
        
        leakage = 0.12 + np.random.normal(0, 0.02)
        leakage = float(np.clip(leakage, 0.0, 0.3))
        
        return alice_key, corrected_bob, leakage
    
    def _cascade_correction(self, alice_key: np.ndarray, bob_key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Simulate Cascade protocol"""
        corrected_alice = alice_key.copy()
        corrected_bob = bob_key.copy()
        
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
                
                p_alice = np.sum(alice_block) % 2
                p_bob = np.sum(bob_block) % 2
                total_leaked += 1
                
                if p_alice != p_bob:
                    error_pos = np.random.randint(len(bob_block))
                    corrected_bob[start + error_pos] ^= 1
                    total_leaked += int(np.log2(block_size))
        
        leakage = total_leaked / len(alice_key) if len(alice_key) > 0 else 0.0
        return corrected_alice, corrected_bob, leakage
