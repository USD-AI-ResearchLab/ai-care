import csv
import os

class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.header_written = os.path.exists(path)

    def log(self, row: dict):
        write_header = not self.header_written

        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if write_header:
                writer.writeheader()
                self.header_written = True

            writer.writerow(row)
