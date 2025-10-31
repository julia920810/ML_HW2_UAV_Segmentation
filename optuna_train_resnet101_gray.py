# optuna_train_resnet101_gray.py
# ==========================================================
# UAV 16-Class Semantic Segmentation (Gray + Sharpen + Rotate/Flip)
# Optuna hyperparameter search + formal training (integrated)
# DeepLabV3-ResNet101 backbone, cosine LR with eta_min>0
# 灰階輸入 + 銳化 + 幾何增強（無天氣域轉換）
# 若 Optuna trial 第3 epoch 沒超過最佳 trial mIoU → 提前中止
# ==========================================================

import os, time, json, random, argparse, platform, math, datetime
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models.segmentation import deeplabv3_resnet101

import albumentations as A
from albumentations.pytorch import ToTensorV2

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

# -------------------------- Constants --------------------------
NUM_CLASSES = 16
global_best_miou = -1.0   # <--- 全域紀錄所有 trial 的最佳 mIoU

# -------------------------- Utils --------------------------
def edge_enhance_block(p=0.6):
    """ 銳化 / 邊緣強化混合 """
    return A.OneOf([
        A.UnsharpMask(blur_limit=(3,7), alpha=(0.7,1.0), p=1.0),
        A.Sharpen(alpha=(0.15,0.35), lightness=(0.9,1.1), p=1.0),
        A.CLAHE(clip_limit=(2.0,4.0), tile_grid_size=(8,8), p=1.0),
        A.Emboss(alpha=(0.1,0.25), strength=(0.2,0.5), p=1.0),
    ], p=p)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

# -------------------------- Dataset --------------------------
class SegDataset(Dataset):
    """ 基本灰階資料集，含旋轉/翻轉/銳化增強 """
    def __init__(self, img_dir, mask_dir=None, img_size=600, train=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.names = sorted([p.name for p in self.img_dir.iterdir()
                             if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        self.train = train
        self.img_size = img_size

        if train:
            self.tfm = A.Compose([
                A.SmallestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomCrop(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, p=0.5),
                edge_enhance_block(p=0.8),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2()
            ])
        else:
            self.tfm = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
                A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
                ToTensorV2()
            ])

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        # === [New] 灰階讀取 ===
        gray = cv2.imread(str(self.img_dir/name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Image not found: {self.img_dir/name}")
        img = np.repeat(gray[..., None], 3, axis=2)  # replicate to 3 channels
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if self.mask_dir is not None:
            mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found for {name}")
            mask = np.clip(mask, 0, NUM_CLASSES-1).astype(np.uint8)
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

def compute_miou(conf_mat):
    diag = np.diag(conf_mat).astype(np.float64)
    union = conf_mat.sum(1) + conf_mat.sum(0) - diag
    iou = diag / np.maximum(union, 1e-8)
    return np.nanmean(iou), iou

@torch.no_grad()
def validate(model, loader, device, num_classes=NUM_CLASSES, return_loss=True):
    model.eval(); conf = np.zeros((num_classes,num_classes), dtype=np.int64)
    ce = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device); masks = masks.to(device)
        logits = model(imgs)["out"]
        total_loss += ce(logits, masks).item()
        pred = logits.argmax(1).cpu().numpy()
        for t,p in zip(masks.cpu().numpy(), pred):
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, _ = compute_miou(conf)
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)
    if return_loss:
        total_loss /= max(len(loader), 1)
        return miou, acc, total_loss
    return miou, acc

# -------------------------- Model --------------------------
try:
    from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
    DEFAULT_W = DeepLabV3_ResNet101_Weights.DEFAULT
except Exception:
    DEFAULT_W = None

def build_model(aux_loss: bool, no_pretrain: bool, device):
    weights = None if no_pretrain else DEFAULT_W
    ctor_aux = True if weights is not None else aux_loss
    model = deeplabv3_resnet101(weights=weights, aux_loss=ctor_aux)
    model.classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model.to(device)

# -------------------------- Loaders --------------------------
def make_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                 img_size, batch_size, seed, num_workers):
    set_seed(seed)
    train_set = SegDataset(train_img_dir, train_mask_dir, img_size=img_size, train=True)
    val_set = SegDataset(val_img_dir, val_mask_dir, img_size=img_size, train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size//2), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# -------------------------- Training helpers --------------------------
def estimate_class_weights(train_dataset, device):
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    for img, mask, _ in loader:
        np_mask = mask.numpy().flatten()
        hist += np.bincount(np_mask, minlength=NUM_CLASSES)
    freq = hist / max(hist.sum(), 1)
    weights = 1 / np.log(1.02 + np.maximum(freq, 1e-12))
    return torch.tensor(weights, dtype=torch.float32, device=device)

def train_one_phase(args, device, train_loader, val_loader,
                    aux_loss, no_pretrain, epochs, save_dir,
                    use_class_weight=True, log_prefix="FORMAL"):
    global global_best_miou

    model = build_model(aux_loss, no_pretrain, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=3e-9)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    if use_class_weight:
        cw = estimate_class_weights(train_loader.dataset, device)
        main_ce = nn.CrossEntropyLoss(weight=cw)
        aux_ce  = nn.CrossEntropyLoss(weight=cw)
    else:
        main_ce = nn.CrossEntropyLoss()
        aux_ce  = nn.CrossEntropyLoss()

    hist_train_loss, hist_val_loss, hist_val_miou, hist_val_acc = [], [], [], []
    best_miou = -1.0
    best_state = None
    log_path = Path(save_dir) / "train_log.txt"
    is_formal = (epochs == args.epochs)

    for epoch in range(1, epochs+1):
        model.train(); t0 = time.time(); run_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device); masks = masks.to(device)
            with torch.cuda.amp.autocast(True):
                out = model(imgs); logits = out["out"]
                loss = main_ce(logits, masks)
                if aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss = loss + 0.4 * aux_ce(out["aux"], masks)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item()

        train_loss = run_loss / max(len(train_loader), 1)
        miou, acc, val_loss = validate(model, val_loader, device, return_loss=True)
        sch.step()

        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_val_miou.append(miou)
        hist_val_acc.append(acc)

        cur_lr = opt.param_groups[0]['lr']
        print(f"[{log_prefix}] Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"mIoU={miou:.4f} acc={acc:.4f} lr={cur_lr:.9f}")

        # === 若是 Optuna trial 且第3 epoch沒有超越全域最佳 => 中止 ===
        if not is_formal and epoch == 3:
            if miou <= global_best_miou:
                print(f"⚠️ Trial early stop: epoch3 mIoU={miou:.4f} ≤ best_trial={global_best_miou:.4f}")
                raise optuna.TrialPruned()

        if miou > best_miou:
            best_miou = miou
            if is_formal:
                best_state = model.state_dict()
                torch.save(best_state, Path(save_dir)/"best.pt")

        # 更新 global_best_miou
        global_best_miou = max(global_best_miou, best_miou)

    # ---------------------------
    # ✅ 訓練結束後再畫圖（三曲線）
    # ---------------------------
    if is_formal:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 左軸：Loss
        ax1.plot(range(1, len(hist_train_loss)+1), hist_train_loss, 'b-', label="Train Loss")
        ax1.plot(range(1, len(hist_val_loss)+1), hist_val_loss, 'r-', label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.3)

        # 右軸：mIoU + Accuracy
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(hist_val_miou)+1), hist_val_miou, 'g-', label="Val mIoU")
        ax2.plot(range(1, len(hist_val_acc)+1), hist_val_acc, 'orange', linestyle='--', label="Val Accuracy")
        ax2.set_ylabel("Metrics", color='g')
        ax2.tick_params(axis='y', labelcolor='g')

        # 合併圖例
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")

        plt.title("Training & Validation Curves (Loss / mIoU / Accuracy)")
        plt.tight_layout()
        plt.savefig(Path(save_dir)/"train_curve.png", dpi=200)
        plt.close(fig)

    return best_miou, best_state, {"val_miou": hist_val_miou}

# -------------------------- Optuna Objective --------------------------
def optuna_objective(trial, base_args, device):
    args = argparse.Namespace(**vars(base_args))
    args.lr = trial.suggest_float("lr", 1e-5, 9e-5, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-8, 5e-6, log=True)
    args.aux_loss = trial.suggest_categorical("aux_loss", [False, True])
    args.no_pretrain = trial.suggest_categorical("no_pretrain", [False, True])

    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size, args.seed, args.num_workers
    )

    best_miou, _, _ = train_one_phase(
        args=args, device=device, train_loader=train_loader, val_loader=val_loader,
        aux_loss=args.aux_loss, no_pretrain=args.no_pretrain,
        epochs=args.proxy_epochs, save_dir=args.save_dir,
        use_class_weight=args.use_class_weight, log_prefix=f"TRIAL{trial.number:02d}"
    )
    return best_miou

# -------------------------- Main --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_img_dir", type=str, default="UAV_dataset/dataset_split/train/imgs")
    p.add_argument("--train_mask_dir", type=str, default="UAV_dataset/dataset_split/train/masks")
    p.add_argument("--val_img_dir", type=str, default="UAV_dataset/dataset_split/val/imgs")
    p.add_argument("--val_mask_dir", type=str, default="UAV_dataset/dataset_split/val/masks")
    p.add_argument("--img_size", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--proxy_epochs", type=int, default=5)
    p.add_argument("--optuna_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0 if platform.system()=="Windows" else 4)
    p.add_argument("--use_class_weight", action="store_true", default=True)
    p.add_argument("--save_root", type=str, default="./outputs_optuna_gray_20251030_0906")
    args = p.parse_args()

    ts = now_tag()
    args.save_dir = str(Path(args.save_root + f"_{ts}").resolve())
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Using device: {device}")
    print(f"==> Output dir : {args.save_dir}")
    # --- Optional: 如果偵測到 best_params.json 就跳過 Optuna ---
    best_json_path = Path(args.save_root) / "best_params.json"
    if best_json_path.exists():
        print(f"==> Found existing best_params.json, skip Optuna search.")
        with open(best_json_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)
    else:
        print("\n==> Starting Optuna hyperparameter search ...")
        sampler = TPESampler(seed=args.seed, multivariate=True)
        pruner  = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(lambda tr: optuna_objective(tr, args, device),
                    n_trials=args.optuna_trials, gc_after_trial=True)

        best_params = study.best_trial.params
        print("\n==> Best params found by Optuna:\n", json.dumps(best_params, indent=2, ensure_ascii=False))
        with open(Path(args.save_dir)/"best_params.json", "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

    print("\n==> Starting FORMAL training with best params ...")
    args.lr = best_params["lr"]
    args.weight_decay = best_params["weight_decay"]
    args.aux_loss = best_params["aux_loss"]
    args.no_pretrain = best_params["no_pretrain"]

    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size, args.seed, args.num_workers
    )

    best_miou, _, _ = train_one_phase(
        args=args, device=device, train_loader=train_loader, val_loader=val_loader,
        aux_loss=args.aux_loss, no_pretrain=args.no_pretrain,
        epochs=args.epochs, save_dir=args.save_dir,
        use_class_weight=args.use_class_weight, log_prefix="FORMAL"
    )

    print(f"\n✅ All done. Best mIoU = {best_miou:.7f}")
    print(f"Artifacts saved in {args.save_dir}")

if __name__ == "__main__":
    main()
