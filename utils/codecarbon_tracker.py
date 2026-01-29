import os
from codecarbon import EmissionsTracker


class CodeCarbonWrapper:
    def __init__(self, project_name="EAPS-CodeCarbon", output_dir="codecarbon_logs"):
        os.makedirs(output_dir, exist_ok=True)   # ✅ FIX

        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            measure_power_secs=1,
            log_level="error",
            tracking_mode="process"
        )

    def __enter__(self):
        self.tracker.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emissions_kg = self.tracker.stop()
