import os
import shutil
import scipy.io as sio


def prepare_imagenet100_val(base_dir, imagenet100_synsets):
    """
    Organize ImageNet-2012 validation images into ImageNet-100 class folders.

    This is a ONE-TIME preprocessing step.
    """

    val_dir = os.path.join(base_dir, "val")
    meta_mat = os.path.join(base_dir, "meta.mat")
    gt_file = os.path.join(base_dir, "ILSVRC2012_validation_ground_truth.txt")

    # ---------------- load meta.mat ----------------
    meta = sio.loadmat(meta_mat)
    synsets = meta["synsets"]

    # class index (1–1000) -> synset id
    idx_to_synset = {
        int(s[0][0][0][0]): s[0][1][0]
        for s in synsets
    }

    # create class folders
    for syn in imagenet100_synsets:
        os.makedirs(os.path.join(val_dir, syn), exist_ok=True)

    # read validation labels
    with open(gt_file, "r") as f:
        labels = [int(x.strip()) for x in f.readlines()]

    moved = 0

    for i, cls_idx in enumerate(labels, start=1):
        syn = idx_to_synset[cls_idx]
        if syn not in imagenet100_synsets:
            continue

        img = f"ILSVRC2012_val_{i:08d}.JPEG"
        src = os.path.join(val_dir, img)
        dst = os.path.join(val_dir, syn, img)

        if os.path.exists(src):
            shutil.move(src, dst)
            moved += 1

    print(f"[ImageNet-100] Validation prepared: {moved} images moved.")


if __name__ == "__main__":
    IMAGENET100 = [
        "n01440764","n01443537","n01484850","n01491361","n01494475",
        "n01514668","n01514859","n01518878","n01530575","n01531178",
        "n01532829","n01534433","n01537544","n01558993","n01560419",
        "n01580077","n01582220","n01592084","n01601694","n01608432",
        "n01614925","n01616318","n01622779","n01630670","n01631663",
        "n01632458","n01632777","n01641577","n01644373","n01644900",
        "n01664065","n01665541","n01667114","n01667778","n01669191",
        "n01675722","n01677366","n01682714","n01685808","n01687978",
        "n01688243","n01689811","n01692333","n01693334","n01694178",
        "n01695060","n01697457","n01698640","n01704323","n01728572",
        "n01728920","n01729322","n01729977","n01734418","n01735189",
        "n01737021","n01739381","n01740131","n01742172","n01744401",
        "n01748264","n01749939","n01751748","n01753488","n01755581",
        "n01756291","n01768244","n01770081","n01770393","n01773157",
        "n01773549","n01773797","n01774384","n01774750","n01775062",
        "n01776313","n01784675","n01795545","n01796340","n01797886",
        "n01798484","n01806143","n01806567","n01807496","n01817953",
        "n01818515","n01819313","n01820546","n01824575","n01828970",
        "n01829413","n01833805","n01843065","n01843383","n01847000"
    ]

    BASE = r"C:\Experiments\eaps\data\imagenet100"
    prepare_imagenet100_val(BASE, IMAGENET100)
