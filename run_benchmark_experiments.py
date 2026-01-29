from utils.config import load_config
from utils.logger import CSVLogger
from utils.carbon_backend import get_carbon_backend

from models.mlp import MLP
from models.cnn import SimpleCNN
from models.tiny_transformer import TinyTransformer
from models.mlp_mixer import MLPMixer
from models.resnet import ResNet18
from models.mobilenet import MobileNetV2
from models.efficientnet import EfficientNetB0

from train import train_model
from evaluate import evaluate_model
from data.vision_datasets import get_dataset, load_imagenet100


# --------------------------------------------------
# Canonical model support table
# --------------------------------------------------
ALLOWED_MODELS = {
    "MNIST": {
        "mlp", "cnn", "transformer", "mlp_mixer", "efficientnet_b0"
    },
    "FASHIONMNIST": {
        "mlp", "cnn", "transformer", "mlp_mixer", "efficientnet_b0"
    },
    "CIFAR10": {
        "mlp", "cnn", "transformer", "mlp_mixer", "efficientnet_b0"
    },
    "CIFAR100": {
        "mlp", "cnn", "transformer", "mlp_mixer", "efficientnet_b0"
    },
    "IMAGENET100": {
        "resnet18", "mobilenetv2", "mlp_mixer", "efficientnet_b0"
    },
}


# --------------------------------------------------
# Auto-cap epochs for fair carbon comparison
# --------------------------------------------------
def auto_cap_epochs(cfg, dataset_key, model_name):
    """
    Prevent compute-heavy models from dominating carbon
    due to excessive epochs on small datasets.
    """
    base_epochs = cfg.get("epochs", 10)

    if model_name == "efficientnet_b0":
        if dataset_key in ["MNIST", "FASHIONMNIST"]:
            return min(base_epochs, 5)
        elif dataset_key in ["CIFAR10", "CIFAR100"]:
            return min(base_epochs, 10)
        elif dataset_key == "IMAGENET100":
            return min(base_epochs, 20)

    return base_epochs


# --------------------------------------------------
# Model builder
# --------------------------------------------------
def build_model(cfg):
    dataset_key = cfg["dataset"].upper().replace("-", "").replace("_", "")
    model_name = cfg["model"]

    if model_name not in ALLOWED_MODELS.get(dataset_key, set()):
        raise ValueError(
            f"Model '{model_name}' is not supported for dataset '{dataset_key}'"
        )

    # ---------------- MLP ----------------
    if model_name == "mlp":
        input_dim = 784 if dataset_key in ["MNIST", "FASHIONMNIST"] else 3072
        return MLP(
            input_dim=input_dim,
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
        )

    # ---------------- CNN ----------------
    elif model_name == "cnn":
        in_ch = 1 if dataset_key in ["MNIST", "FASHIONMNIST"] else 3
        return SimpleCNN(
            in_channels=in_ch,
            channels=cfg["channels"],
            num_classes=cfg["num_classes"],
            dataset=dataset_key,
        )

    # ---------------- Tiny Transformer ----------------
    elif model_name == "transformer":
        input_dim = 784 if dataset_key in ["MNIST", "FASHIONMNIST"] else 3072
        return TinyTransformer(
            input_dim=input_dim,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            layers=cfg["layers"],
            num_classes=cfg["num_classes"],
        )

    # ---------------- MLP-Mixer ----------------
    elif model_name == "mlp_mixer":
        in_ch = 1 if dataset_key in ["MNIST", "FASHIONMNIST"] else 3

        if dataset_key in ["MNIST", "FASHIONMNIST"]:
            image_size = 28
        elif dataset_key in ["CIFAR10", "CIFAR100"]:
            image_size = 32
        else:  # IMAGENET100
            image_size = 224

        return MLPMixer(
            image_size=image_size,
            patch_size=cfg["patch_size"],
            in_channels=in_ch,
            num_classes=cfg["num_classes"],
            hidden_dim=cfg["hidden_dim"],
            depth=cfg["depth"],
            token_dim=cfg["token_dim"],
            channel_dim=cfg["channel_dim"],
        )

    # ---------------- ResNet-18 ----------------
    elif model_name == "resnet18":
        return ResNet18(num_classes=cfg["num_classes"])

    # ---------------- MobileNetV2 ----------------
    elif model_name == "mobilenetv2":
        return MobileNetV2(num_classes=cfg["num_classes"])

    # ---------------- EfficientNet-B0 ----------------
    elif model_name == "efficientnet_b0":
        return EfficientNetB0(num_classes=cfg["num_classes"])

    else:
        raise ValueError(f"Unknown model: {model_name}")


# --------------------------------------------------
# Main experiment loop
# --------------------------------------------------
def main():

    logger = CSVLogger("results/results.csv")

    datasets = [
        "mnist",
        "fashion_mnist",
        "cifar10",
        "cifar100",
        "imagenet100",
    ]

    models = [
        "mlp",
        "cnn",
        "transformer",
        "mlp_mixer",
        "efficientnet_b0",
        "resnet18",
        "mobilenetv2",
    ]

    for dataset in datasets:
        for model in models:

            try:
                cfg = load_config([
                    "configs/base.yaml",
                    f"configs/{dataset}.yaml",
                    f"configs/{model}.yaml",
                ])

                dataset_key = cfg["dataset"].upper().replace("-", "").replace("_", "")

                # ---- Auto-cap epochs (fair carbon) ----
                cfg["epochs"] = auto_cap_epochs(cfg, dataset_key, cfg["model"])

                # ---------------- Dataset loading ----------------
                if dataset_key == "IMAGENET100":
                    train_set, test_set, meta = load_imagenet100(log=True)
                    cfg["num_classes"] = meta["train_classes"]
                else:
                    train_set = get_dataset(cfg["dataset"], train=True)
                    test_set = get_dataset(cfg["dataset"], train=False)
                    meta = {
                        "dataset": cfg["dataset"],
                        "train_classes": len(train_set.classes),
                        "val_classes": len(test_set.classes),
                    }

                model_instance = build_model(cfg)

            except (ValueError, FileNotFoundError) as e:
                print(f"[SKIP] {dataset} + {model} → {e}")
                continue

            # ---------------- Training ----------------
            if cfg.get("train", True):
                model_instance, train_kwh = train_model(
                    model_instance, train_set, cfg
                )
                train_carbon = get_carbon_backend(cfg).compute_carbon(
                    train_kwh, phase="train"
                )
            else:
                train_kwh = 0.0
                train_carbon = 0.0

            # ---------------- Evaluation ----------------
            eval_out = evaluate_model(model_instance, test_set, cfg)

            # Backward-safe unpacking
            if len(eval_out) == 3:
                acc, infer_kwh, val_classes = eval_out
            else:
                acc, infer_kwh = eval_out
                val_classes = meta["val_classes"]

            infer_carbon = get_carbon_backend(cfg).compute_carbon(
                infer_kwh, phase="infer"
            )

            # ---------------- Logging ----------------
            logger.log({
                "dataset": cfg["dataset"],
                "model": cfg["model"],
                "accuracy": acc,
                "train_kwh": train_kwh,
                "train_carbon": train_carbon,
                "infer_kwh": infer_kwh,
                "inference_carbon": infer_carbon,
                "train_classes": meta["train_classes"],
                "val_classes": val_classes,
            })


if __name__ == "__main__":
    main()
