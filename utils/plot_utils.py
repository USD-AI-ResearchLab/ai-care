import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def _clean_csv(df):
    # Remove accidental repeated header rows
    if "model" in df.columns:
        df = df[df["model"] != "model"]

    # -------------------------------
    # Backward compatibility:
    # Old CSVs used `carbon` for inference-time carbon
    # New schema uses `inference_carbon`
    # -------------------------------
    if "carbon" in df.columns and "inference_carbon" not in df.columns:
        df["inference_carbon"] = df["carbon"]

    return df



def _canonical_dataset(name: str) -> str:
    """
    Normalize dataset names:
    ImageNet100, ImageNet-100, imagenet_100 → imagenet100
    """
    return name.lower().replace("-", "").replace("_", "")


def _val_class_note(sub_df):
    """
    Returns a short validation-class note if val_classes is present
    and differs from train_classes.
    """
    if "val_classes" not in sub_df.columns:
        return None

    vc = sub_df["val_classes"].dropna().unique()
    tc = sub_df.get("train_classes", pd.Series()).dropna().unique()

    if len(vc) == 1:
        if len(tc) == 1 and vc[0] != tc[0]:
            return f"Validation classes = {int(vc[0])}"
        if len(tc) == 0:
            return f"Validation classes = {int(vc[0])}"

    return None


def pareto_front(df, x_col="inference_carbon", y_col="accuracy"):
    """
    Boolean mask of Pareto-optimal points.
    Minimize inference carbon, maximize accuracy.
    """
    x = df[x_col].values
    y = df[y_col].values

    is_pareto = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        if not is_pareto[i]:
            continue
        dominates = (
            (x <= x[i]) &
            (y >= y[i]) &
            ((x < x[i]) | (y > y[i]))
        )
        if dominates.any():
            is_pareto[i] = False

    return is_pareto


# --------------------------------------------------
# Model families (canonical)
# --------------------------------------------------
MODEL_FAMILIES = {
    "CNN": {"cnn", "resnet18", "mobilenetv2"},
    "Transformer": {"transformer"},
    "Mixer": {"mlp_mixer"},
}


# --------------------------------------------------
# Export Pareto tables
# --------------------------------------------------
def export_pareto_table(csv_path, out_dir):
    df = _clean_csv(pd.read_csv(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    for dataset_key in df["dataset"].apply(_canonical_dataset).unique():
        sub_df = df[df["dataset"].apply(_canonical_dataset) == dataset_key].copy()

        pareto_df = sub_df[
            pareto_front(sub_df)
        ].sort_values(
            by=["inference_carbon", "accuracy"],
            ascending=[True, False]
        )

        out_path = os.path.join(out_dir, f"pareto_{dataset_key}.csv")
        pareto_df.to_csv(out_path, index=False)
        print(f"[Pareto] Exported {len(pareto_df)} points → {out_path}")


# --------------------------------------------------
# Global accuracy–carbon overview
# --------------------------------------------------
def plot_catc(csv_path, out_path):
    df = _clean_csv(pd.read_csv(csv_path))

    plt.figure(figsize=(6.5, 5.5))

    dataset_markers = {
        "mnist": "o",
        "fashionmnist": "s",
        "cifar10": "^",
        "cifar100": "D",
        "imagenet100": "X",
    }

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        for dataset_key in model_df["dataset"].apply(_canonical_dataset).unique():
            sub = model_df[
                model_df["dataset"].apply(_canonical_dataset) == dataset_key
            ]
            plt.scatter(
                sub["inference_carbon"],
                sub["accuracy"],
                marker=dataset_markers.get(dataset_key, "o"),
                s=60,
                alpha=0.8,
                label=f"{model} ({dataset_key})",
            )

    note = _val_class_note(df[df["dataset"].apply(_canonical_dataset) == "imagenet100"])
    if note:
        plt.text(0.99, 0.01, f"ImageNet-100: {note}",
                 transform=plt.gca().transAxes,
                 ha="right", va="bottom", fontsize=9)

    plt.xlabel("Inference Carbon (g CO$_2$)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy–Carbon Overview Across Datasets")
    plt.legend(fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


# --------------------------------------------------
# CATC per dataset
# --------------------------------------------------
def plot_catc_by_dataset(csv_path, out_dir):
    df = _clean_csv(pd.read_csv(csv_path))
    os.makedirs(out_dir, exist_ok=True)

    for dataset_key in df["dataset"].apply(_canonical_dataset).unique():
        sub_df = df[df["dataset"].apply(_canonical_dataset) == dataset_key]

        pareto_mask = pareto_front(sub_df)

        plt.figure(figsize=(6, 5))
        plt.scatter(
            sub_df.loc[~pareto_mask, "inference_carbon"],
            sub_df.loc[~pareto_mask, "accuracy"],
            s=60, alpha=0.35, color="gray", label="Dominated"
        )

        pareto_df = sub_df[pareto_mask]
        plt.scatter(
            pareto_df["inference_carbon"],
            pareto_df["accuracy"],
            s=120, edgecolors="black", linewidths=1.5, label="Pareto-optimal"
        )

        if len(pareto_df) > 1:
            pareto_df = pareto_df.sort_values("inference_carbon")
            plt.plot(pareto_df["inference_carbon"], pareto_df["accuracy"], "--")

        plt.xlabel("Inference Carbon (g CO$_2$)")
        plt.ylabel("Accuracy")

        title = f"Carbon–Accuracy Tradeoff Curve (CATC): {dataset_key}"
        note = _val_class_note(sub_df)
        if note:
            title += f"\n({note})"

        plt.title(title)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"catc_{dataset_key}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()


# --------------------------------------------------
# CARE-AI efficiency
# --------------------------------------------------
def plot_care_ai_efficiency(csv_path, out_dir):
    df = _clean_csv(pd.read_csv(csv_path))

    df["inference_carbon"] = pd.to_numeric(df["inference_carbon"], errors="coerce")
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")
    df = df.dropna(subset=["accuracy", "inference_carbon"])
    df = df[df["inference_carbon"] > 0]

    df["care_ai_efficiency"] = df["accuracy"] / df["inference_carbon"]

    os.makedirs(out_dir, exist_ok=True)

    for dataset_key in df["dataset"].apply(_canonical_dataset).unique():
        sub_df = df[df["dataset"].apply(_canonical_dataset) == dataset_key]

        agg = (
            sub_df.groupby("model", as_index=False)
            .agg(care_ai_efficiency=("care_ai_efficiency", "mean"))
            .sort_values("care_ai_efficiency", ascending=False)
        )

        plt.figure(figsize=(7, 4.5))
        plt.bar(agg["model"], agg["care_ai_efficiency"])
        plt.ylabel("CARE-AI Efficiency (Accuracy / Carbon)")
        plt.title(f"CARE-AI Efficiency: {dataset_key}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"care_ai_efficiency_{dataset_key}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()


# --------------------------------------------------
# Carbon consistency (EAPS vs CodeCarbon)
# --------------------------------------------------
def plot_carbon_consistency_eaps_vs_codecarbon(
    eaps_csv, codecarbon_csv, out_dir
):
    os.makedirs(out_dir, exist_ok=True)

    eaps = pd.read_csv(eaps_csv)[["dataset", "model", "inference_carbon"]].rename(
        columns={"inference_carbon": "eaps_carbon"}
    )
    cc = pd.read_csv(codecarbon_csv)[["dataset", "model", "carbon"]].rename(
        columns={"carbon": "cc_carbon"}
    )

    merged = pd.merge(eaps, cc, on=["dataset", "model"], how="inner")
    merged = merged.dropna()

    x = merged["eaps_carbon"].values
    y = merged["cc_carbon"].values
    rho, _ = spearmanr(x, y)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=80)
    min_v, max_v = min(x.min(), y.min()), max(x.max(), y.max())
    plt.plot([min_v, max_v], [min_v, max_v], "--", color="gray")

    plt.xlabel("Inference Carbon (EAPS)")
    plt.ylabel("Inference Carbon (CodeCarbon)")
    plt.title(f"Global Carbon Consistency (ρ={rho:.2f})")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "carbon_consistency_eaps_vs_codecarbon.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
