# ============================================================
# 16 類語義分割 (UAV) — Optuna 超參搜尋 + 正式訓練 + 測試
# ============================================================

import os, time, json, random, argparse, platform
from pathlib import Path
import numpy as np
import cv2
import gc
cv2.setNumThreads(0)
import sys
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import pandas as pd
import random
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from torch.utils.data import Dataset, DataLoader
import numpy as np
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

NUM_CLASSES = 16

# -------------------------- Utils --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def edge_enhance_block(p=0.6):
    return A.OneOf([
        A.UnsharpMask(blur_limit=(3,7), alpha=(0.7,1.0), p=1.0),
        A.Sharpen(alpha=(0.15,0.35), lightness=(0.9,1.1), p=1.0),
        A.CLAHE(clip_limit=(2.0,4.0), tile_grid_size=(8,8), p=1.0),
        A.Emboss(alpha=(0.1,0.25), strength=(0.2,0.5), p=1.0),
    ], p=p)

# -------------------------- Dataset --------------------------
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=768, train=True,
                 augment_pack="weather", fixed_tfm: A.Compose=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.names = sorted([p.name for p in self.img_dir.iterdir()
                             if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        self.train = train; self.img_size = img_size

        if fixed_tfm is not None:
            self.tfm = fixed_tfm
        elif train:
            if augment_pack == "stronger":
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomResizedCrop(img_size, img_size, scale=(0.55,1.25), ratio=(0.85,1.15), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),
                    A.Perspective(scale=(0.02,0.08), p=0.3),
                    A.CoarseDropout(max_holes=10, max_height=32, max_width=32, p=0.3),
                            # ----- 天氣域轉換 -----
                    # p=0.6：有 60% 機率做「某種天氣效果」，
                    # 其中包含 Normal（等於不動），所以不會每張都被破壞到爆。
                    A.OneOf([
                        # Normal（保持原樣，當作一種 domain）
                        A.NoOp(p=1.0),

                        # Rain
                        A.RandomRain(
                            slant_lower=-10, slant_upper=10,
                            drop_length=20, drop_width=1,
                            blur_value=3,
                            brightness_coefficient=0.95,
                            p=1.0
                        ),

                        # Snow
                        A.RandomSnow(
                            snow_point_lower=0.1, snow_point_upper=0.3,
                            brightness_coeff=1.5,
                            p=1.0
                        ),

                        # Fallen：模擬秋天/落葉/偏黃＋局部暗影
                        A.Compose([
                            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2,
                                        shadow_dimension=5, p=1.0),
                            A.ColorJitter(brightness=0.2, contrast=0.2,
                                        saturation=0.3, hue=0.05, p=1.0),
                        ], p=1.0),

                        # Dust：偏黃霧、對比下降
                        A.Compose([
                            A.RandomFog(fog_coef_lower=0.02, fog_coef_upper=0.10,
                                        alpha_coef=0.08, p=1.0),
                            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1),
                                                    contrast_limit=(-0.3, 0.0),
                                                    p=1.0),
                        ], p=1.0),

                        # Fog：濃霧版
                        A.RandomFog(
                            fog_coef_lower=0.15, fog_coef_upper=0.35,
                            alpha_coef=0.12,
                            p=1.0
                        ),
                    ], p=1.0),

                    # ----- 顏色 / 質感 & 正規化 -----
                    A.RandomBrightnessContrast(p=0.6),
                    A.ColorJitter(p=0.5),
                    edge_enhance_block(p=0.8),
                    A.ImageCompression(60,90,p=0.3),
                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
                ])
            else:  # edge
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.6),
                    A.RandomBrightnessContrast(p=0.5),
                    edge_enhance_block(p=0.6),
                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
                ])
        else:
            self.tfm = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
            ])

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(str(self.img_dir/name))[:,:,::-1]
        img = cv2.resize(img, (self.img_size, self.img_size))
        if self.mask_dir:
            mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            out = self.tfm(image=img, mask=mask)
            return out["image"], out["mask"].long(), name
        else:
            out = self.tfm(image=img)
            return out["image"], name

# -------------------------- Metrics --------------------------
def fast_hist(true, pred, num_classes):
    k = (true >= 0) & (true < num_classes)
    return np.bincount(num_classes * true[k].astype(int) + pred[k],
                       minlength=num_classes**2).reshape(num_classes, num_classes)

def compute_miou_per_class(conf_mat):
    #根據混淆矩陣計算每類別 IoU 與平均 IoU
    diag = np.diag(conf_mat).astype(np.float64)
    union = conf_mat.sum(1) + conf_mat.sum(0) - diag
    iou_per_class = diag / np.maximum(union, 1e-8)
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou, iou_per_class

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    val_loss = 0.0
    ce = nn.CrossEntropyLoss().to(device)

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        out = model(imgs)["out"]
        val_loss += ce(out, masks).item()
        pred = out.argmax(1).cpu().numpy()
        for t, p in zip(masks.cpu().numpy(), pred):
            conf += fast_hist(t.flatten(), p.flatten(), NUM_CLASSES)

    mean_iou, iou_per_class = compute_miou_per_class(conf)
    acc = np.diag(conf).sum() / conf.sum()
    val_loss /= max(len(loader), 1)

    # 🔹 顯示各類別 IoU
    print("\nPer-Class IoU:")
    for cid, iou in enumerate(iou_per_class):
        print(f"  class_{cid:02d}: {iou:.4f}")
    print(f"  ==> mIoU: {mean_iou:.4f}, Acc: {acc:.4f}, Loss: {val_loss:.4f}\n")

    return mean_iou, acc, val_loss


# -------------------------- Model --------------------------
def build_model(aux_loss, no_pretrain, device):
    use_w = None if no_pretrain else DeepLabV3_ResNet50_Weights.DEFAULT
    # 修正：如果有 weights，就強制 aux_loss=True，避免 ValueError
    if use_w is not None and aux_loss is False:
        print("Warning: torchvision requires aux_loss=True when pretrained weights are used. Forcing aux_loss=True.")
        aux_loss = True
    m = deeplabv3_resnet50(weights=use_w, aux_loss=aux_loss)
    m.classifier[-1] = nn.Conv2d(256, NUM_CLASSES, 1)
    if getattr(m, "aux_classifier", None): m.aux_classifier[-1] = nn.Conv2d(256, NUM_CLASSES, 1)
    return m.to(device)

# -------------------------- Loader Builder --------------------------
def make_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                 img_size, batch_size, seed, num_workers, augment_pack="weather",
                 include_rotflip=False, include_sharpen=False,
                 ):
    set_seed(seed)

    # 主訓練集
    train_main = SegDataset(train_img_dir, train_mask_dir,
                            img_size=img_size, train=True, augment_pack=augment_pack)

    # 旋轉+翻轉：固定版（每張都做，等同新增一份資料）
    train_rf = None
    if include_rotflip:
        rf_tfm = A.Compose([
            A.SmallestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
        ])
        train_rf = SegDataset(train_img_dir, train_mask_dir,
                              img_size=img_size, train=True, fixed_tfm=rf_tfm)

    # 銳化：固定版（每張都做）
    train_sh = None
    if include_sharpen:
        sh_tfm = A.Compose([
            A.SmallestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            edge_enhance_block(p=1.0),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
        ])
        train_sh = SegDataset(train_img_dir, train_mask_dir,
                              img_size=img_size, train=True, fixed_tfm=sh_tfm)

    # 合併所有可用訓練集
    train_datasets = [d for d in [train_main, train_rf, train_sh] if d is not None]
    train_set = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)

    val_set = SegDataset(val_img_dir, val_mask_dir, img_size=img_size, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size//2), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# -------------------------- Optuna Objective --------------------------
def objective(trial, base_args):
    if platform.system() == "Windows" and base_args.num_workers != 0:
        base_args.num_workers = 0

    # 複製 base args
    args = argparse.Namespace(**vars(base_args))

    # Optuna 搜尋空間（調整後）
    args.lr = trial.suggest_float("lr", 0.005, 0.012, log=True)  # 縮小範圍在 0.0087 附近
    args.weight_decay = trial.suggest_float("weight_decay", 1e-4, 8e-4, log=True)  # 集中 4e-4 區域
    args.momentum = trial.suggest_float("momentum", 0.85, 0.90)  # 收斂於 0.87
    args.img_size = trial.suggest_categorical("img_size", [600])  # 固定 600
    args.batch_size = trial.suggest_categorical("batch_size", [4,8])  # 既然可行就固定
    args.aux_loss = trial.suggest_categorical("aux_loss", [True])  # 最佳時是 True，固定即可
    args.no_pretrain = trial.suggest_categorical("no_pretrain", [False,True])  # 最佳時是 False
    args.augment_pack = trial.suggest_categorical("augment_pack", ["edge"])  # 兩者仍可比較


    # 🔹 把本次 trial 的參數印進 log（會同時出現在螢幕 + run_xxx.log）
    print("\n" + "="*60)
    print(f"[Trial {trial.number}] Params:")
    print(json.dumps({
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "aux_loss": args.aux_loss,
        "no_pretrain": args.no_pretrain,
        "augment_pack": args.augment_pack,
    }, indent=2))
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size, args.seed, args.num_workers, args.augment_pack,
        include_rotflip=args.include_rotflip,
        include_sharpen=args.include_sharpen,
    )

    # ---------- class weight ----------
    if args.use_class_weight:
        print("==> Estimating class weights ...")
        hist = np.zeros(NUM_CLASSES)
        subset = torch.utils.data.Subset(train_loader.dataset, range(min(200, len(train_loader.dataset))))
        for img, mask, _ in DataLoader(subset, batch_size=1, shuffle=False):
            np_mask = mask.numpy().flatten()
            hist += np.bincount(np_mask, minlength=NUM_CLASSES)
        freq = hist / hist.sum()
        weights = 1/np.log(1.01+freq)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        print("Class weights:", np.round(weights.cpu().numpy(), 3))
        main_ce = nn.CrossEntropyLoss(weight=weights)
        aux_ce  = nn.CrossEntropyLoss(weight=weights)
    else:
        main_ce = nn.CrossEntropyLoss()
        aux_ce  = nn.CrossEntropyLoss()

    model = build_model(args.aux_loss, args.no_pretrain, device)

    opt = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    T = base_args.proxy_epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T, eta_min=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_miou = -1.0

    for epoch in range(1, T + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(True):
                out = model(imgs)
                logits = out["out"]
                loss = main_ce(logits, masks)
                if args.aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss = loss + 0.4 * aux_ce(out["aux"], masks)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()

        miou, acc, vloss = validate(model, val_loader, device)
        sch.step()
        trial.report(miou, step=epoch)

        # ======================================================
        # 🔹【跨-trial 第 3 epoch 比較規則】
        # ======================================================
        if epoch == 3:
            current = float(miou)
            best_epoch3 = trial.study.user_attrs.get("best_epoch3_miou", None)

            if (best_epoch3 is None) or (current > best_epoch3):
                trial.study.set_user_attr("best_epoch3_miou", current)
                print(f"[Epoch3] Trial {trial.number} sets new best_epoch3_miou = {current:.4f}")
            else:
                print(f"[Cross-Trial Prune] Trial {trial.number} pruned at epoch 3: "
                      f"{current:.4f} < best_epoch3_miou {best_epoch3:.4f}")
                raise optuna.TrialPruned()

        # ======================================================
        # 🔹【trial 內部 early stopping】
        # ======================================================
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
        print(f"[Epoch {epoch:03d}] train_loss={run_loss/max(len(train_loader),1):.4f}  "
              f"val_loss={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}  "
              f"lr={opt.param_groups[0]['lr']:.6f}  time={(time.time()-t0):.1f}s")

    print(f"✅ Trial {trial.number} finished, best mIoU = {best_miou:.4f}")

    torch.cuda.empty_cache()
    gc.collect()

    return float(best_miou)


# -------------------------- 正式訓練 --------------------------
def train_best_params(args, best_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 優先讀取 best_params.json（確保一致）
    best_path = Path(args.save_dir) / "best_params.json"
    if best_path.exists():
        with open(best_path, "r") as f:
            best_params = json.load(f)
        aux_loss = best_params.get("aux_loss", True)
        no_pretrain = best_params.get("no_pretrain", False)
        augment_pack = best_params.get("augment_pack", "edge")   
        print(f"==> Using best params for training: aux_loss={aux_loss}, no_pretrain={no_pretrain}")
    else:
        aux_loss, no_pretrain = True, False
        augment_pack = "edge"
        print("⚠️ No best_params.json found, using default aux_loss=True, no_pretrain=False")

    # 控制 batch size，避免 OOM
    bs_opt = int(best_params.get("batch_size", args.batch_size))
    if bs_opt > 4:
        print("⚠️ Batch size too large for your GPU, forcing batch_size=4 to avoid OOM.")
        bs_opt = 4
    args.batch_size = bs_opt
    # 先建立 Train / Val Loader 
    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size,
        args.seed, args.num_workers,
        augment_pack,
        include_rotflip=args.include_rotflip
    )

    # -----------------------------------------
    # 2) 計算 class weight（如果有啟用）
    # -----------------------------------------
    if args.use_class_weight:
        print("==> Estimating class weights for FINAL training ...")
        class_hist = np.zeros(NUM_CLASSES, dtype=np.float64)

        subset_size = min(3000, len(train_loader.dataset))
        subset = torch.utils.data.Subset(train_loader.dataset, range(subset_size))
        tmp_loader = DataLoader(subset, batch_size=1, shuffle=False)

        for _, mask, _ in tmp_loader:
            np_mask = mask.numpy().ravel()
            class_hist += np.bincount(np_mask, minlength=NUM_CLASSES)

        freq = class_hist / class_hist.sum()
        weights = 1.0 / np.log(1.01 + freq)
        weights = weights / weights.mean()
        weights_t = torch.tensor(weights, dtype=torch.float32).to(device)
        print("Class weights:", np.round(weights, 3))

        ce_main = nn.CrossEntropyLoss(weight=weights_t)
        ce_aux  = nn.CrossEntropyLoss(weight=weights_t)
    else:
        ce_main = nn.CrossEntropyLoss()
        ce_aux  = nn.CrossEntropyLoss()

    # -----------------------------------------
    # 3) 建立模型
    # -----------------------------------------
    model = build_model(aux_loss=aux_loss, no_pretrain=no_pretrain, device=device)

    # -----------------------------------------
    # 4) Optimizer
    # -----------------------------------------
    lr           = float(best_params.get("lr", 0.001))
    momentum     = float(best_params.get("momentum", 0.9))
    weight_decay = float(best_params.get("weight_decay", 1e-4))

    opt = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=args.epochs,   
    eta_min=1e-6        # 最小 lr
    )

    # -----------------------------------------
    # 5) 開始訓練
    # -----------------------------------------
    patience = 6
    no_improve = 0
    best_miou = -1.0
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "miou": [], "acc": []}
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.cuda.amp.autocast(True):
                out = model(imgs)
                loss = ce_main(out["out"], masks)

                if aux_loss and "aux" in out and out["aux"] is not None:
                    loss = loss + 0.4 * ce_aux(out["aux"], masks)

                

            # ✅ 混合精度 backward
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()

        miou, acc, vloss = validate(model, val_loader, device)
        scheduler.step()
        print(f"current lr = {scheduler.get_last_lr()[0]:.6f}")

        hist["epoch"].append(ep)
        hist["train_loss"].append(run_loss / len(train_loader))
        hist["val_loss"].append(vloss)
        hist["miou"].append(miou)
        hist["acc"].append(acc)

        print(f"[FINAL] Epoch {ep:03d} train loss={run_loss/len(train_loader):.4f} val loss ={vloss:.4f} mIoU={miou:.4f} time={(time.time()-t0):.1f}s")

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
            print(f"  --> [SAVE] new best_miou={best_miou:.4f}, saved to {args.save_dir}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {ep}, best mIoU={best_miou:.4f}")
                break

    # ✅ 儲存結果圖與紀錄
    df = pd.DataFrame(hist)
    df.to_csv(Path(args.save_dir) / "metrics.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.legend(); plt.grid()
    plt.savefig(Path(args.save_dir) / "loss_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["miou"], label="mIoU")
    plt.plot(df["epoch"], df["acc"], label="Acc")
    plt.legend(); plt.grid()
    plt.savefig(Path(args.save_dir) / "miou_acc_curve.png", dpi=150)
    plt.close()


# -------------------------- 測試 --------------------------
@torch.no_grad()
def test_best_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔹 載入 Optuna 找到的最佳組合
    best_path = Path(args.save_dir) / "best_params.json"
    if best_path.exists():
        with open(best_path, "r") as f:
            best_params = json.load(f)
        aux_loss = best_params.get("aux_loss", True)
        no_pretrain = best_params.get("no_pretrain", False)
        print(f"==> Using best params for test: aux_loss={aux_loss}, no_pretrain={no_pretrain}")
    else:
        # 若沒找到 best_params.json，就退回預設
        aux_loss, no_pretrain = True, False
        print("⚠️ No best_params.json found, using default aux_loss=True, no_pretrain=False")

    # ✅ 根據最佳組合建模
    model = build_model(aux_loss=aux_loss, no_pretrain=no_pretrain, device=device)

    # ✅ 載入最佳權重
    state = torch.load(Path(args.save_dir) / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 測試資料集
    test_set = SegDataset(args.test_img_dir, None, args.img_size, train=False)
    test_loader = DataLoader(test_set, batch_size=1)

    out_dir = Path(args.save_dir) / "preds"
    out_dir.mkdir(exist_ok=True)
    csv_path = Path(args.save_dir) / "sample_submission.csv"

    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["img"] + [f"class_{i}" for i in range(NUM_CLASSES)])
        for img, name in test_loader:
            img = img.to(device)
            logits = model(img)["out"]
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(out_dir / f"{Path(name[0]).stem}.png"), pred)

            row = [name[0]]
            for cid in range(NUM_CLASSES):
                mask = (pred == cid).astype(np.uint8)
                if mask.sum() == 0:
                    row.append("none")
                else:
                    pixels = mask.flatten(order="F")
                    pixels = np.concatenate([[0], pixels, [0]])
                    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
                    runs[1::2] -= runs[:-1:2]
                    row.append(" ".join(str(x) for x in runs))
            writer.writerow(row)

    print("✅ Test complete:", out_dir)



class FocusClassDataset(Dataset):
    """
    將原本的 segmentation dataset 包起來：
    - 只要某張 mask 裡出現 focus_classes（預設 14,15）
      就把這張的 index 重複 factor 次（例如 3 倍）
    - 其他圖片維持一次
    """
    def __init__(self, base_ds, focus_classes=(14, 15), factor=4):
        self.base_ds = base_ds
        self.focus_classes = focus_classes
        self.factor = factor

        base_indices = list(range(len(base_ds)))
        focus_idx = []

        print(f"==> Scanning dataset for focus classes {focus_classes} ...")
        for i in range(len(base_ds)):
            sample = base_ds[i]
            # 你的 dataset 回傳 (img, mask, meta) 或 (img, mask)
            if len(sample) == 3:
                _, mask, _ = sample
            else:
                _, mask = sample

            mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
            if any((mask_np == c).any() for c in focus_classes):
                focus_idx.append(i)

        print(f"==> Found {len(focus_idx)} samples containing {focus_classes}")

        # 原本所有樣本各 1 次 + focus 样本多 (factor-1) 次
        self.indices = base_indices + focus_idx * (factor - 1)
        print(f"==> Original size = {len(base_ds)}, oversampled size = {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_ds[real_idx]





class FocusClassPatchDataset(Dataset):
    """
    將原本 segmentation dataset 包起來：
    - 有 focus_classes (預設 14,15) 的樣本會被 oversample (factor 倍)
    - 其中一部分樣本會回傳「含 14/15 的局部 patch」，其餘回傳 full image
    """
    def __init__(self, base_ds,
                 focus_classes=(14, 15),
                 factor=3,
                 patch_prob=0.5,
                 patch_ratio=0.6):
        """
        base_ds: 你原本的 SegDataset
        factor: oversample 倍數（有 14/15 的圖被重複幾倍）
        patch_prob: 這張 sample 會用 patch 模式的機率
        patch_ratio: patch 寬高 / full image（例如 0.6 表示 600→360）
        """
        self.base_ds = base_ds
        self.focus_classes = focus_classes
        self.factor = factor
        self.patch_prob = patch_prob
        self.patch_ratio = patch_ratio

        # 從 base_ds 抓必要資訊
        self.img_dir = base_ds.img_dir      # Path
        self.mask_dir = base_ds.mask_dir    # Path
        self.names = base_ds.names          # list of filenames
        self.img_size = base_ds.img_size    # e.g. 600
        self.tfm = base_ds.tfm              # Albumentations transform

        base_indices = list(range(len(base_ds)))
        focus_idx = []

        print(f"==> [FocusClassPatchDataset] Scanning dataset for focus classes {focus_classes} ...")
        for i in range(len(base_ds)):
            # 用原本的 __getitem__ 取 mask，或直接重新讀檔都可以
            sample = base_ds[i]
            if len(sample) == 3:
                _, mask, _ = sample
            else:
                _, mask = sample

            mask_np = mask.numpy() if hasattr(mask, "numpy") else np.array(mask)
            if any((mask_np == c).any() for c in focus_classes):
                focus_idx.append(i)

        print(f"==> [FocusClassPatchDataset] Found {len(focus_idx)} focus samples.")
        # 所有樣本各1次 + focus 樣本多 (factor-1) 次
        self.indices = base_indices + focus_idx * (factor - 1)
        print(f"==> [FocusClassPatchDataset] Original size={len(base_ds)}, oversampled size={len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        name = self.names[real_idx]

        # 重新讀圖 & mask（跟你的 SegDataset 一樣流程）
        img_path = self.img_dir / name
        mask_path = self.mask_dir / name

        img = cv2.imread(str(img_path))[:, :, ::-1]  # BGR->RGB
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # resize 到固定大小
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        mask_np = mask  # 已經是 numpy

        has_focus = any((mask_np == c).any() for c in self.focus_classes)

        # 預設先用 full image
        crop_img = img
        crop_mask = mask

        # 若這張含 14/15 & 且抽到 patch 模式 → 截一塊 patch
        if has_focus and (random.random() < self.patch_prob):
            # 找所有 focus pixel 的座標
            ys, xs = np.where(
                np.logical_or.reduce([(mask_np == c) for c in self.focus_classes])
            )
            if len(xs) > 0:
                # 隨機選一個 focus pixel 當中心
                k = random.randint(0, len(xs) - 1)
                cx, cy = xs[k], ys[k]

                H, W = mask_np.shape
                patch_size = int(self.img_size * self.patch_ratio)
                patch_size = max(32, min(patch_size, self.img_size))  # 安全值

                half = patch_size // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(W, x1 + patch_size)
                y2 = min(H, y1 + patch_size)

                # 若剛好靠邊，重新對齊
                x1 = max(0, x2 - patch_size)
                y1 = max(0, y2 - patch_size)

                crop_img = img[y1:y2, x1:x2, :]
                crop_mask = mask[y1:y2, x1:x2]

                # 再 resize 回原本輸入大小
                crop_img = cv2.resize(crop_img, (self.img_size, self.img_size))
                crop_mask = cv2.resize(crop_mask, (self.img_size, self.img_size),
                                       interpolation=cv2.INTER_NEAREST)

        # 丟進 Albumentations
        out = self.tfm(image=crop_img, mask=crop_mask)
        img_t = out["image"]
        mask_t = out["mask"].long()

        return img_t, mask_t, name

# -------------------------- 第二階段 fine-tune --------------------------
# def fine_tune_best(args, extra_epochs=30):
#     """
#     第二階段 fine-tune：
#     - 從第一階段的 best.pt 繼續訓練
#     - 開啟 class weight，特別 boost 小類別
#     - lr 用第一階段的 1/3
#     """
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("\n================ Fine-tune Stage ================")
#     print("==> Device:", device)

#     # 1) 讀取 best_params.json，拿到原本最佳超參
#     best_path = Path(args.save_dir) / "best_params.json"
#     if not best_path.exists():
#         print("❌ 找不到 best_params.json，無法 fine-tune")
#         return

#     with open(best_path, "r") as f:
#         best_params = json.load(f)

#     aux_loss    = best_params.get("aux_loss", True)
#     no_pretrain = best_params.get("no_pretrain", False)
#     base_lr     = float(best_params.get("lr", 0.001))
#     momentum    = float(best_params.get("momentum", 0.9))
#     weight_decay = float(best_params.get("weight_decay", 1e-4))
#     augment_pack = best_params.get("augment_pack", "edge")
#     bs_opt      = int(best_params.get("batch_size", args.batch_size))

#     if bs_opt > 4:
#         bs_opt = 4
#     args.batch_size = bs_opt

#     print(f"==> best_params (for fine-tune):")
#     print(f"    aux_loss={aux_loss}, no_pretrain={no_pretrain}")
#     print(f"    base_lr={base_lr}, momentum={momentum}, weight_decay={weight_decay}")
#     print(f"    augment_pack={augment_pack}, batch_size={bs_opt}")

#     # 2) 建立模型並載入第一階段的 best.pt
#     model = build_model(aux_loss=aux_loss, no_pretrain=no_pretrain, device=device)

#     ckpt_path = Path(args.save_dir) / "best.pt"
#     if not ckpt_path.exists():
#         print("❌ 找不到 best.pt，請先完成第一階段訓練")
#         return

#     state = torch.load(ckpt_path, map_location="cpu")
#     model.load_state_dict(state, strict=False)
#     model.to(device)
#     model.eval()

#     # 3) DataLoader：跟第一階段一樣的 augment_pack
#     train_loader, val_loader = make_loaders(
#         args.train_img_dir, args.train_mask_dir,
#         args.val_img_dir,   args.val_mask_dir,
#         args.img_size, args.batch_size,
#         args.seed, args.num_workers,
#         augment_pack,
#         include_rotflip=args.include_rotflip,
#         include_sharpen=args.include_sharpen,
#     )

#     # 4) 計算 class weight（使用訓練集），並提升小類別權重
#     print("==> Computing class weights for fine-tune ...")
#     hist = np.zeros(NUM_CLASSES)
#     # 用部分 sample（例如 400 張）估計就好，避免太慢
#     subset = torch.utils.data.Subset(train_loader.dataset,
#                                      range(min(400, len(train_loader.dataset))))
#     sub_loader = DataLoader(subset, batch_size=1, shuffle=False)
#     for _, masks, _ in sub_loader:
#         np_mask = masks.numpy().flatten()
#         hist += np.bincount(np_mask, minlength=NUM_CLASSES)

#     freq = hist / np.maximum(hist.sum(), 1)
#     weights = 1.0 / np.log(1.02 + freq)  # baseline
#     print("class_weights (before boost):", np.round(weights, 3))
#     # 針對較難的小物件加強（你可以依自己的 class 再調整）
#     # mid_ids = [5, 9, 12]
#     # for cid in mid_ids:
#     #     weights[cid] *= 1.1
#     hard_ids = [14, 15]
#     for cid in hard_ids:
#         weights[cid] *= 1.2   # 若覺得太兇可以改 1.3
#     weights = weights / weights.mean()  # 可選，但通常我會加

#     print("class_weights (after boost):", np.round(weights, 3))
#     w_t = torch.tensor(weights, dtype=torch.float32).to(device)
#     ce_main = nn.CrossEntropyLoss(weight=w_t)
#     ce_aux  = nn.CrossEntropyLoss(weight=w_t)

#     # 5) optimizer：lr 用原本的 1/3 做 fine-tune
#     ft_lr = base_lr * 0.1
#     print(f"==> Fine-tune lr = {ft_lr:.6f}")
#     # 先凍結 backbone
#     for p in model.backbone.parameters():
#         p.requires_grad = False

#     # 再開啟 classifier / aux_classifier
#     for p in model.classifier.parameters():
#         p.requires_grad = True

#     if hasattr(model, "aux_classifier"):
#         for p in model.aux_classifier.parameters():
#             p.requires_grad = True

#     opt = torch.optim.SGD(
#         model.parameters(),
#         lr=ft_lr,
#         momentum=momentum,
#         weight_decay=weight_decay,
#         nesterov=True,
#     )

#     scaler = torch.cuda.amp.GradScaler(enabled=True)

#     # 先算一下「目前未 fine-tune 的 mIoU」，當作 baseline
#     print("==> Evaluate baseline (before fine-tune) ...")
#     base_miou, base_acc, _ = validate(model, val_loader, device)
#     best_miou = base_miou
#     print(f"Baseline mIoU = {base_miou:.4f}, Acc = {base_acc:.4f}")
#     hist = {"epoch": [], "train_loss": [], "val_loss": [], "miou": [], "acc": []}
#     # 6) 開始 fine-tune
#     for ep in range(1, extra_epochs + 1):
#         model.train()
#         run_loss = 0.0
#         t0 = time.time()

#         for imgs, masks, _ in train_loader:
#             imgs, masks = imgs.to(device), masks.to(device)

#             with torch.cuda.amp.autocast(True):
#                 out = model(imgs)
#                 logits = out["out"]
#                 loss = ce_main(logits, masks)

#                 # 🔹 若最佳參數啟用 aux_loss，且模型真的有 aux，就一併算
#                 if aux_loss and ("aux" in out) and (out["aux"] is not None):
#                     loss = loss + 0.4 * ce_aux(out["aux"], masks)

#             opt.zero_grad(set_to_none=True)
#             scaler.scale(loss).backward()
#             scaler.step(opt)
#             scaler.update()

#             run_loss += loss.item()

#         miou, acc, vloss = validate(model, val_loader, device)

#         print(f"[FT] Epoch {ep:03d}  train={run_loss/len(train_loader):.4f}  "
#               f"val={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}  "
#               f"time={(time.time()-t0):.1f}s")
#         hist["epoch"].append(ep)
#         hist["train_loss"].append(run_loss / len(train_loader))
#         hist["val_loss"].append(vloss)
#         hist["miou"].append(miou)
#         hist["acc"].append(acc)
#         # 若 fine-tune 後的 mIoU 比 baseline 好，就覆蓋 best.pt
#         if miou > best_miou:
#             best_miou = miou
#             torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
#             print(f"  ✅ New best mIoU {best_miou:.4f}, overwrite best.pt")

#     print(f"==> Fine-tune done. Best mIoU (including baseline) = {best_miou:.4f}")
def fine_tune_best(args, extra_epochs=8, oversample_factor=4):
    """
    第二階段 fine-tune（資料分佈版）：
    - 從第一階段的 best.pt 繼續訓練
    - 不用 class weight，用含 14/15 的影像 oversample
    - lr 用第一階段的 0.1 倍（較溫和）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n================ Fine-tune Stage (oversample 14/15) ================")
    print("==> Device:", device)

    # 1) 讀取 best_params.json，拿到原本最佳超參
    best_path = Path(args.save_dir) / "best_params.json"
    if not best_path.exists():
        print("❌ 找不到 best_params.json，無法 fine-tune")
        return

    with open(best_path, "r") as f:
        best_params = json.load(f)

    aux_loss     = best_params.get("aux_loss", True)
    no_pretrain  = best_params.get("no_pretrain", False)
    base_lr      = float(best_params.get("lr", 0.001))
    momentum     = float(best_params.get("momentum", 0.9))
    weight_decay = float(best_params.get("weight_decay", 1e-4))
    augment_pack = best_params.get("augment_pack", "edge")
    bs_opt       = int(best_params.get("batch_size", args.batch_size))

    if bs_opt > 4:
        bs_opt = 4
    args.batch_size = bs_opt

    print("==> best_params (for fine-tune):")
    print(f"    aux_loss={aux_loss}, no_pretrain={no_pretrain}")
    print(f"    base_lr={base_lr}, momentum={momentum}, weight_decay={weight_decay}")
    print(f"    augment_pack={augment_pack}, batch_size={bs_opt}")

    # 2) 建立模型並載入第一階段的 best.pt
    model = build_model(aux_loss=aux_loss, no_pretrain=no_pretrain, device=device)

    ckpt_path = Path(args.save_dir) / "best.pt"
    if not ckpt_path.exists():
        print("❌ 找不到 best.pt，請先完成第一階段訓練")
        return

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)

    # 3) DataLoader：先照第一階段一樣建立，再對 train 做 oversample
    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir,   args.val_mask_dir,
        args.img_size, args.batch_size,
        args.seed, args.num_workers,
        augment_pack,
        include_rotflip=args.include_rotflip,
        #include_sharpen=args.include_sharpen,
    )

    base_train_ds = train_loader.dataset
    focus_patch_ds = FocusClassPatchDataset(
        base_train_ds,
        focus_classes=(14, 15),
        factor=oversample_factor,   # 建議先用 3
        patch_prob=0.5,             # 50% 機率用 patch
        patch_ratio=0.6             # patch 大約是 600*0.6=360
    )

    train_loader = DataLoader(
        #focus_patch_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # 🔹用 FocusClassDataset 包起來（只對 train 做 oversample）
    # base_train_ds = train_loader.dataset
    # focus_ds = FocusClassDataset(base_train_ds, focus_classes=(14, 15),
    #                              factor=oversample_factor)
    # train_loader = DataLoader(
    #     focus_ds,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )

    # 4) 不用 class weight，改回普通 CrossEntropyLoss
    ce_main = nn.CrossEntropyLoss()
    ce_aux  = nn.CrossEntropyLoss()
    print("==> Using plain CrossEntropyLoss (no class weight), with oversampled 14/15.")

    # 5) optimizer：lr 用原本的 0.1 倍做 fine-tune
    ft_lr = base_lr * 0.1
    print(f"==> Fine-tune lr = {ft_lr:.6f}")

    # 凍結 backbone，只訓練 classifier / aux_classifier
    for p in model.backbone.parameters():
        p.requires_grad = False

    for p in model.classifier.parameters():
        p.requires_grad = True

    if hasattr(model, "aux_classifier") and (model.aux_classifier is not None):
        for p in model.aux_classifier.parameters():
            p.requires_grad = True

    opt = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=ft_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # 6) baseline：未 fine-tune 的 mIoU
    print("==> Evaluate baseline (before fine-tune) ...")
    base_miou, base_acc, _ = validate(model, val_loader, device)
    best_miou = base_miou
    print(f"Baseline mIoU = {base_miou:.4f}, Acc = {base_acc:.4f}")

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "miou": [], "acc": []}

    # 7) 開始 fine-tune
    for ep in range(1, extra_epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.cuda.amp.autocast(True):
                out = model(imgs)
                logits = out["out"]
                loss = ce_main(logits, masks)

                if aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss = loss + 0.4 * ce_aux(out["aux"], masks)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()

        miou, acc, vloss = validate(model, val_loader, device)

        print(f"[FT-OS] Epoch {ep:03d}  train={run_loss/len(train_loader):.4f}  "
              f"val={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}  "
              f"time={(time.time()-t0):.1f}s")

        hist["epoch"].append(ep)
        hist["train_loss"].append(run_loss / len(train_loader))
        hist["val_loss"].append(vloss)
        hist["miou"].append(miou)
        hist["acc"].append(acc)

        # 只有當 fine-tune 後 mIoU 比 baseline 好，才覆蓋 best.pt
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), Path(args.save_dir) / "best.pt")
            print(f"  ✅ New best mIoU {best_miou:.4f}, overwrite best.pt")

    print(f"==> Fine-tune done. Best mIoU (including baseline) = {best_miou:.4f}")

    # ✅ 儲存結果圖與紀錄
    df = pd.DataFrame(hist)
    df.to_csv(Path(args.save_dir) / "metrics_fine_tune.csv", index=False)

    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train")
    plt.plot(df["epoch"], df["val_loss"], label="val")
    plt.legend(); plt.grid()
    plt.savefig(Path(args.save_dir) / "loss_curve_fine_tune.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(df["epoch"], df["miou"], label="mIoU")
    plt.plot(df["epoch"], df["acc"], label="Acc")
    plt.legend(); plt.grid()
    plt.savefig(Path(args.save_dir) / "miou_acc_curve_fine_tune.png", dpi=150)
    plt.close()
def fine_tune_best1(args, extra_epochs=8):
    """
    第二階段 fine-tune（保守版，專門救泛化）：
    - 從第一階段 outputs_optuna_final_1110/best.pt 繼續訓練
    - 不再用 class weight（避免把模型壓得太貼 train/val 分佈）
    - 凍結 backbone，只更新 classifier / aux_classifier
    - lr = 第一次訓練 lr 的 0.1 倍左右
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n================ Fine-tune Stage (seed123 model) ================")
    print("==> Device:", device)

    save_dir = Path(args.save_dir)

    # 1) 讀 best_params.json
    best_path = save_dir / "best_params.json"
    if not best_path.exists():
        print("❌ 找不到 best_params.json，沒辦法對齊第一階段設定")
        return

    with open(best_path, "r") as f:
        best_params = json.load(f)

    aux_loss    = best_params.get("aux_loss", True)
    no_pretrain = best_params.get("no_pretrain", False)
    base_lr     = float(best_params.get("lr", 0.001))
    momentum    = float(best_params.get("momentum", 0.9))
    weight_decay = float(best_params.get("weight_decay", 1e-4))
    augment_pack = best_params.get("augment_pack", "edge")
    bs_opt      = int(best_params.get("batch_size", args.batch_size))

    if bs_opt > 4:
        bs_opt = 4
    args.batch_size = bs_opt

    print("==> fine-tune from:")
    print(f"    aux_loss={aux_loss}, no_pretrain={no_pretrain}")
    print(f"    base_lr={base_lr}, momentum={momentum}, weight_decay={weight_decay}")
    print(f"    augment_pack={augment_pack}, batch_size={bs_opt}")

    # 2) 建立模型 & 載入第一階段 best.pt（就是你 seed 123 那一版）
    ckpt_path = save_dir / "best_0.7237.pt"
    if not ckpt_path.exists():
        print(f"❌ 找不到 {ckpt_path}，請先跑完第一階段訓練")
        return

    model = build_model(aux_loss=aux_loss, no_pretrain=no_pretrain, device=device)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)

    # ⚠️ 非常重要：先算 baseline mIoU，當作 fine-tune 的比較基準
    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir,   args.val_mask_dir,
        args.img_size, args.batch_size,
        args.seed, args.num_workers,
        augment_pack,
        include_rotflip=args.include_rotflip
    )

    print("==> Evaluate baseline (before fine-tune) ...")
    base_miou, base_acc, base_loss = validate(model, val_loader, device)
    print(f"Baseline mIoU = {base_miou:.4f}, Acc = {base_acc:.4f}, Loss = {base_loss:.4f}")

    # 3) 不再用 class weight，回到最單純的 CE
    ce_main = nn.CrossEntropyLoss()
    ce_aux  = nn.CrossEntropyLoss()

    # 4) 凍結 backbone，只微調 classifier / aux_classifier
    for p in model.backbone.parameters():
        p.requires_grad = False

    trainable_params = []
    for p in model.classifier.parameters():
        p.requires_grad = True
        trainable_params.append(p)

    if hasattr(model, "aux_classifier") and (model.aux_classifier is not None):
        for p in model.aux_classifier.parameters():
            p.requires_grad = True
            trainable_params.append(p)

    # 5) Optimizer：用較小 lr（避免破壞原本已經學好的東西）
    ft_lr = base_lr * 0.2      # 例如 0.0087 * 0.2 ≒ 0.0017
    print(f"==> Fine-tune lr = {ft_lr:.6f}")

    opt = torch.optim.SGD(
        trainable_params,
        lr=ft_lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_miou = base_miou
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "miou": [], "acc": []}

    for ep in range(1, extra_epochs + 1):
        model.train()
        run_loss = 0.0
        t0 = time.time()

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.cuda.amp.autocast(True):
                out = model(imgs)
                logits = out["out"]
                loss = ce_main(logits, masks)

                if aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss = loss + 0.4 * ce_aux(out["aux"], masks)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            run_loss += loss.item()

        miou, acc, vloss = validate(model, val_loader, device)

        print(f"[FT] Epoch {ep:03d}  train={run_loss/len(train_loader):.4f}  "
              f"val={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}  "
              f"time={(time.time()-t0):.1f}s")

        hist["epoch"].append(ep)
        hist["train_loss"].append(run_loss / len(train_loader))
        hist["val_loss"].append(vloss)
        hist["miou"].append(miou)
        hist["acc"].append(acc)
        new_path = save_dir / f"best_new_{best_miou:.4f}.pt"
        # 只要有比 baseline mIoU 好，就直接覆蓋 best.pt
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), new_path)
            print(f"  ✅ New best mIoU {best_miou:.4f}, overwrite best.pt")

    print(f"==> Fine-tune done. Best mIoU (including baseline) = {best_miou:.4f}")

# -------------------------- Main --------------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train_img_dir",default="UAV_dataset/dataset_split/train/imgs")
    p.add_argument("--train_mask_dir",default="UAV_dataset/dataset_split/train/masks")
    p.add_argument("--val_img_dir",default="UAV_dataset/dataset_split/val/imgs")
    p.add_argument("--val_mask_dir",default="UAV_dataset/dataset_split/val/masks")
    p.add_argument("--test_img_dir",default="UAV_dataset/test")
    p.add_argument("--img_size",type=int,default=600)
    p.add_argument("--batch_size",type=int,default=8)
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--proxy_epochs",type=int,default=5)
    p.add_argument("--optuna_trials",type=int,default=10)
    p.add_argument("--save_dir",default="./outputs_optuna_final_1110")
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--aux_loss", action="store_true")
    p.add_argument("--augment_pack", type=str,default="stronger",choices=["stronger", "edge"])
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--num_workers",type=int,default=0)
    # 是否將「旋轉翻轉版」與「銳化版」當新樣本併入（線上生成，不寫檔）
    p.add_argument("--include_rotflip", action="store_true")
    p.add_argument("--include_sharpen", action="store_true")
    p.add_argument("--use_class_weight", action="store_true",help="若開啟則依據類別比例自動計算權重 (改善資料不平衡)")
    p.add_argument("--skip_optuna", action="store_true",
                   help="若設定此參數，則直接載入 best_params.json 進行訓練，不再執行搜尋")
    p.add_argument("--finetune", action="store_true",
                   help="啟用第二階段 fine-tune（從 best.pt 續訓）")
    p.add_argument("--ft_epochs", type=int, default=30,
                   help="fine-tune 的 epoch 數（預設 25）")
    args=p.parse_args()
    os.makedirs(args.save_dir,exist_ok=True)

     # 🔹 啟用自動 log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.save_dir) / f"run_{timestamp}.log"
    log_f = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print(f"==> Logging to {log_path}")

    set_seed(args.seed)
    
    if args.finetune:
        print("==> Finetune-only mode (跳過第一階段訓練與 Optuna)")
        fine_tune_best1(args)
        print("\n==> Testing best ft model ...")
        # 用 fine-tune 出來的 best_ft.pt 測試，你可以改成讀 best_ft.pt
        test_best_model(args)
        return
    
    if args.skip_optuna:
        best_path = Path(args.save_dir) / "best_params.json"
        if not best_path.exists():
            print("❌ 找不到 best_params.json，請先跑過 Optuna 搜尋。")
            return
        print("✅ 跳過 Optuna，直接載入最佳參數訓練模型 ...")
        with open(best_path, "r") as f:
            best_params = json.load(f)

        train_best_params(args, best_params)
        print("\n==> Testing best model ...")
        test_best_model(args)
        return

    print("==> Starting Optuna Search...")
    study=optuna.create_study(direction="maximize",sampler=TPESampler(seed=args.seed),
                              pruner=SuccessiveHalvingPruner())
    study.optimize(lambda tr:objective(tr,args),n_trials=args.optuna_trials)
    print("==> Best mIoU:",study.best_value)
    print(json.dumps(study.best_trial.params,indent=2))
    json.dump(study.best_trial.params,open(Path(args.save_dir)/"best_params.json","w"),indent=2)
    # 🔹 儲存所有 trial 的結果（包含參數、value、state 等）
    df_trials = study.trials_dataframe()
    df_trials.to_csv(Path(args.save_dir) / "optuna_trials.csv", index=False)
    print("==> All trial results saved to optuna_trials.csv")

    print("\n==> Training final model ...")
    train_best_params(args,study.best_trial.params)
    print("\n==> Testing best model ...")
    test_best_model(args)

if __name__=="__main__":
    main()