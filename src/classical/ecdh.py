try:
    from Crypto.PublicKey import ECC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from src.qkd.device import ResourceLimitedDevice

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
        
        if not self.device.execute_task("ECDH KeyGen", 4, int(5e7), 50):
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
        
        point = peer_public_key.pointQ * self.private_key.d
        self.shared_secret = point.x.to_bytes(32, 'big')
        return True
