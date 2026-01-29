import time
import psutil


class EnergyMeter:
    """
    Measures energy consumption only.
    Carbon computation is handled by carbon_backend.
    """

    def __init__(self, device: str):
        self.device = device
        self.energy_j = 0.0
        self.start_time = None
        self.process = psutil.Process()

    def start(self):
        self.start_time = time.time()
        self.start_cpu_energy = self._read_cpu_energy()

    def stop(self):
        end_cpu_energy = self._read_cpu_energy()
        self.energy_j += max(0.0, end_cpu_energy - self.start_cpu_energy)

    def add_flops(self, flops: float, mem_access_bytes: float):
        """
        Optional: approximate dynamic energy contribution.
        This is a heuristic and intentionally backend-agnostic.
        """
        # You can extend this later if needed
        pass

    def report(self):
        """
        Return energy consumption in kWh.
        """
        kwh = self.energy_j / 3.6e6
        return kwh

    def _read_cpu_energy(self):
        """
        Approximate CPU energy using time and utilization.
        This is a proxy; exact measurement is backend-dependent.
        """
        cpu_percent = self.process.cpu_percent(interval=None) / 100.0
        elapsed = time.time() - self.start_time if self.start_time else 0.0

        # Conservative estimate: 65W TDP CPU
        cpu_power_watts = 65.0
        return cpu_percent * cpu_power_watts * elapsed
