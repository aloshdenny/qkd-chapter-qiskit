import time

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
        
        if not self.allocate_ram(ram_bytes):
            print(f"{self.name}: Insufficient RAM for {name} ({ram_kb}KB needed, {(self.ram_limit - self.allocated_ram)//1024}KB available)")
            return False
        
        if power_mw > self.power_limit:
            print(f"{self.name}: Power limit exceeded for {name} ({power_mw}mW > {self.power_limit}mW)")
            self.free_ram(ram_bytes)
            return False
        
        execution_time = cpu_cycles / (self.cpu_freq * 1e6)
        time.sleep(min(execution_time, 0.01))
        
        self.total_cpu_cycles += cpu_cycles
        self.total_power_consumed += power_mw * execution_time / 1000
        
        self.free_ram(ram_bytes)
        return True
