import os
from pathlib import Path
import cv2
import numpy as np
import random
import shutil

# ===== 依你的專案改這裡 =====
IMG_DIR = r"UAV_dataset/train/imgs"
MASK_DIR = r"UAV_dataset/train/masks"
OUT_ROOT = r"dataset_split_ratio"   # 新版輸出資料夾
NUM_CLASSES = 16
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
COPY_FILES = True
# =============================


def read_mask_counts(mask_path, num_classes=16):
    """讀取一張遮罩，回傳各類像素數 array=(num_classes,)"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"讀不到遮罩: {mask_path}")
    counts = np.bincount(mask.flatten(), minlength=num_classes)
    return counts


def score_split(current_counts, add_counts, target_counts):
    """
    給定目前某個 split 的 class_counts，加上這張圖的 counts 之後，
    算它跟目標 target_counts 的「偏離程度」。
    這裡用 L1（絕對值和）當作簡單的距離。
    """
    new_counts = current_counts + add_counts
    diff = np.abs(new_counts - target_counts)
    return diff.sum(), new_counts


def main():
    random.seed(RANDOM_SEED)

    img_dir = Path(IMG_DIR)
    mask_dir = Path(MASK_DIR)
    out_root = Path(OUT_ROOT)

    # 1) 收集圖片 + 遮罩
    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    pairs = []
    for img_path in img_paths:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            raise RuntimeError(f"找不到對應遮罩: {mask_path}")
        pairs.append((img_path, mask_path))

    total_images = len(pairs)
    print(f"[INFO] 找到 {total_images} 張圖")

    # 2) 先掃一遍算「全資料集」的 class 總像素
    per_image = []
    global_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for img_path, mask_path in pairs:
        counts = read_mask_counts(mask_path, NUM_CLASSES)
        global_class_counts += counts
        present = np.where(counts > 0)[0].tolist()
        per_image.append({
            "img": img_path,
            "mask": mask_path,
            "counts": counts,
            "present": present
        })

    print("[INFO] 全資料集像素分布：")
    for c in range(NUM_CLASSES):
        print(f"  class {c:02d}: {global_class_counts[c]} px")

    # 3) 算每一類的「目標 train 像素數」
    target_train_counts = (global_class_counts * TRAIN_RATIO).astype(np.int64)
    target_val_counts = global_class_counts - target_train_counts

    # 4) 為了讓稀有類別先被好好分配，還是先照「稀有度」排一下順序
    for info in per_image:
        rarity = 0.0
        for c in info["present"]:
            rarity += 1.0 / (global_class_counts[c] + 1)
        info["rarity"] = rarity

    # 稀有的先處理
    per_image.sort(key=lambda x: x["rarity"], reverse=True)

    # 目標張數
    target_train_imgs = int(total_images * TRAIN_RATIO)
    target_val_imgs = total_images - target_train_imgs

    # 真正的 split 結果
    train_infos = []
    val_infos = []

    # 動態記錄目前兩邊的 class 像素量
    train_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    val_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for info in per_image:
        counts = info["counts"]
        present = info["present"]

        # 如果 train 還沒看過這個類別，而這張圖有 → 優先丟 train
        need_train_for_coverage = False
        for c in present:
            if train_class_counts[c] == 0:
                need_train_for_coverage = True
                break

        # 如果 train 還沒滿 → 可以考慮 train
        train_not_full = len(train_infos) < target_train_imgs
        val_not_full = len(val_infos) < target_val_imgs

        # 情況一：一定要給 train（為了 coverage）
        if need_train_for_coverage and train_not_full:
            train_infos.append(info)
            train_class_counts += counts
            continue

        # 情況二：其中一邊已經滿了
        if not train_not_full and val_not_full:
            # train 滿了，只能給 val
            val_infos.append(info)
            val_class_counts += counts
            continue
        if not val_not_full and train_not_full:
            # val 滿了，只能給 train
            train_infos.append(info)
            train_class_counts += counts
            continue

        # 情況三：兩邊都還有空位 → 算「丟哪邊比較像 8:2」
        if train_not_full and val_not_full:
            # 試丟 train
            train_score, tmp_train_counts = score_split(train_class_counts, counts, target_train_counts)
            # 試丟 val
            val_score, tmp_val_counts = score_split(val_class_counts, counts, target_val_counts)

            # 比較誰的偏離比較小
            if train_score <= val_score:
                train_infos.append(info)
                train_class_counts = tmp_train_counts
            else:
                val_infos.append(info)
                val_class_counts = tmp_val_counts

    # 5) 輸出結果
    print(f"[RESULT] train: {len(train_infos)} 張, val: {len(val_infos)} 張")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train_imgs.txt").write_text("\n".join([str(x["img"]) for x in train_infos]), encoding="utf-8")
    (out_root / "val_imgs.txt").write_text("\n".join([str(x["img"]) for x in val_infos]), encoding="utf-8")

    if COPY_FILES:
        train_img_dir = out_root / "train" / "imgs"
        train_mask_dir = out_root / "train" / "masks"
        val_img_dir = out_root / "val" / "imgs"
        val_mask_dir = out_root / "val" / "masks"
        for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for info in train_infos:
            shutil.copy2(info["img"], train_img_dir / info["img"].name)
            shutil.copy2(info["mask"], train_mask_dir / info["mask"].name)
        for info in val_infos:
            shutil.copy2(info["img"], val_img_dir / info["img"].name)
            shutil.copy2(info["mask"], val_mask_dir / info["mask"].name)

        print(f"[INFO] 已複製到 {out_root.resolve()}")

    # 6) 印分布讓你確認
    print("\n[TRAIN] class pixel counts:")
    for c in range(NUM_CLASSES):
        print(f"  class {c:02d}: {train_class_counts[c]} px  (target: {target_train_counts[c]})")

    print("\n[VAL] class pixel counts:")
    for c in range(NUM_CLASSES):
        print(f"  class {c:02d}: {val_class_counts[c]} px  (target: {target_val_counts[c]})")

    # 檢查 coverage
    missing_in_train = []
    for c in range(NUM_CLASSES):
        if train_class_counts[c] == 0 and val_class_counts[c] > 0:
            missing_in_train.append(c)

    if missing_in_train:
        print("\n[WARN] 還是有類別 val 有但 train 沒有 →", missing_in_train)
        print("你可以手動把這幾張圖移回 train")
    else:
        print("\n[OK] coverage 成功 ✅")


if __name__ == "__main__":
    main()
