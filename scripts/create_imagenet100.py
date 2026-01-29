import os
import shutil

# --------------------------------------------------
# Resolve paths robustly
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# CHANGE THIS to your actual ImageNet-1K root
# Example (Windows):
# IMAGENET_ROOT = r"D:\datasets\ILSVRC2012"
#
# Example (Linux):
# IMAGENET_ROOT = "/data/imagenet/ILSVRC2012"

IMAGENET_ROOT = r"C:\path\to\ILSVRC2012"

OUTPUT_ROOT = os.path.join("data", "imagenet100")
CLASS_LIST = os.path.join(SCRIPT_DIR, "imagenet100_classes.txt")

# --------------------------------------------------
# Create output directories
# --------------------------------------------------
for split in ["train", "val"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, split), exist_ok=True)

# --------------------------------------------------
# Load class list
# --------------------------------------------------
with open(CLASS_LIST, "r") as f:
    classes = [c.strip() for c in f if c.strip()]

print(f"Preparing ImageNet-100 with {len(classes)} classes")

# --------------------------------------------------
# Copy data
# --------------------------------------------------
for split in ["train", "val"]:
    for cls in classes:
        src = os.path.join(IMAGENET_ROOT, split, cls)
        dst = os.path.join(OUTPUT_ROOT, split, cls)

        if not os.path.exists(src):
            print(f"[WARN] Missing {src}")
            continue

        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied {split}/{cls}")
