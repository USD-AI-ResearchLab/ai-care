# utils/carbon_backend.py

class CarbonBackend:
    def compute_carbon(self, energy_kwh, phase):
        raise NotImplementedError


class FixedCarbonBackend(CarbonBackend):
    def __init__(self, intensity_g_per_kwh: float):
        self.intensity = float(intensity_g_per_kwh)

    def compute_carbon(self, energy_kwh, phase):
        return float(energy_kwh) * self.intensity


class CodeCarbonBackend(CarbonBackend):
    def __init__(self, tracker):
        self.tracker = tracker

    def compute_carbon(self, energy_kwh, phase):
        """
        Return total emissions in grams CO2.
        Must be called AFTER tracker.stop().
        """
        # Version-robust access
        if hasattr(self.tracker, "final_emissions"):
            emissions_kg = self.tracker.final_emissions

        elif hasattr(self.tracker, "final_emissions_data"):
            emissions_kg = self.tracker.final_emissions_data.emissions

        else:
            raise RuntimeError(
                "CodeCarbon tracker stopped but emissions attribute not found"
            )

        return float(emissions_kg) * 1000.0


def get_carbon_backend(cfg, tracker=None):
    """
    Factory for carbon backend.
    """
    backend_cfg = cfg.get("carbon_backend", {})
    name = backend_cfg.get("name", "fixed").lower()

    if name == "fixed":
        intensity = backend_cfg.get(
            "intensity_g_per_kwh", 400.0
        )
        return FixedCarbonBackend(intensity)

    if name == "codecarbon":
        if tracker is None:
            raise RuntimeError(
                "CodeCarbon backend requires a finalized tracker instance"
            )
        return CodeCarbonBackend(tracker)

    raise ValueError(f"Unknown carbon backend: {name}")
