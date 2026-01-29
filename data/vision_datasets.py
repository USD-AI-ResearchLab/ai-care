import os
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

# --------------------------------------------------
# Global paths
# --------------------------------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")

# --------------------------------------------------
# ImageNet normalization
# --------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --------------------------------------------------
# ImageNet-100 transforms
# --------------------------------------------------
imagenet_train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

imagenet_val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ==================================================
# Generic dataset loader (kept for compatibility)
# ==================================================
def get_dataset(name, train=True):
    name = name.lower()
    transform = transforms.ToTensor()

    if name == "mnist":
        return datasets.MNIST(DATA_DIR, train=train, download=True, transform=transform)

    if name == "fashion-mnist":
        return datasets.FashionMNIST(DATA_DIR, train=train, download=True, transform=transform)

    if name == "cifar10":
        return datasets.CIFAR10(DATA_DIR, train=train, download=True, transform=transform)

    if name == "cifar100":
        return datasets.CIFAR100(DATA_DIR, train=train, download=True, transform=transform)

    if name == "imagenet100":
        split = "train" if train else "val"
        transform = imagenet_train_transform if train else imagenet_val_transform
        root = os.path.join(DATA_DIR, "imagenet100", split)
        return ImageFolder(root=root, transform=transform)

    raise ValueError(f"Unknown dataset: {name}")


# ==================================================
# ImageNet-100 loader with explicit metadata + logging
# ==================================================
def load_imagenet100(
    root=os.path.join(DATA_DIR, "imagenet100"),
    image_size=224,
    log=True,
):
    train_dir = os.path.join(root, "train")
    val_dir   = os.path.join(root, "val")

    train_set = ImageFolder(train_dir, transform=imagenet_train_transform)
    val_set   = ImageFolder(val_dir,   transform=imagenet_val_transform)

    n_train_classes = len(train_set.classes)
    n_val_classes   = len(val_set.classes)

    # ---------------- logging ----------------
    if log:
        print(f"[ImageNet-100] Train classes: {n_train_classes}")
        if n_val_classes < n_train_classes:
            print(
                f"[ImageNet-100] Val classes: {n_val_classes} "
                f"(missing {n_train_classes - n_val_classes} synsets "
                f"in ImageNet-2012 val)"
            )
        else:
            print(f"[ImageNet-100] Val classes: {n_val_classes}")

        # Safety check
        assert set(val_set.classes).issubset(
            set(train_set.classes)
        ), "Validation classes must be subset of train classes"

    # ---------------- metadata ----------------
    meta = {
        "dataset": "ImageNet-100",
        "train_classes": n_train_classes,
        "val_classes": n_val_classes,
    }

    return train_set, val_set, meta
