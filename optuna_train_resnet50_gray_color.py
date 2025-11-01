# ==========================================================
# UAV 16-Class Semantic Segmentation (Color + Gray + Weather + Sharpen)
# Optuna hyperparameter search + formal training (integrated)
# DeepLabV3-ResNet50 backbone, cosine LR with eta_min>0
# 彩色輸入 + 邊緣銳化 + 幾何增強 + 天氣域轉換（雨、雪、霧、灰塵）+ 隨機灰階
# 每張圖都有「原圖 + 線上增強版」雙倍資料
# ==========================================================

import os, time, json, random, argparse, platform, datetime
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(0)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

import albumentations as A
from albumentations.pytorch import ToTensorV2

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

# -------------------------- Constants --------------------------
NUM_CLASSES = 16
global_best_miou = -1.0

# -------------------------- Utils --------------------------
def edge_enhance_block(p=0.6):
    return A.OneOf([
        A.UnsharpMask(blur_limit=(3,7), alpha=(0.7,1.0), p=1.0),
        A.Sharpen(alpha=(0.15,0.35), lightness=(0.9,1.1), p=1.0),
        A.CLAHE(clip_limit=(2.0,4.0), tile_grid_size=(8,8), p=1.0),
        A.Emboss(alpha=(0.1,0.25), strength=(0.2,0.5), p=1.0),
    ], p=p)

def now_tag():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------------- Augmentations --------------------------
def get_weather_gray_aug(p=0.4):
    """灰階 or 天氣域轉換"""
    return A.OneOf([
        A.ToGray(p=1.0),
        A.OneOf([
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, brightness_coefficient=0.9, p=1.0),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=1.0, p=1.0),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.25, alpha_coef=0.05, p=1.0),
            A.Compose([
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.12, alpha_coef=0.03, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=-20, val_shift_limit=10, p=1.0),
            ], p=1.0),
        ], p=1.0)
    ], p=p)

def get_train_transform(img_size):
    return A.Compose([
        A.SmallestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                           border_mode=cv2.BORDER_CONSTANT, p=0.5),
        edge_enhance_block(p=0.7),
        get_weather_gray_aug(p=0.4),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def get_base_transform(img_size):
    return A.Compose([
        A.SmallestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.RandomCrop(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=10,
                           border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})


def get_val_transform(img_size):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        A.Resize(img_size, img_size, interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

# -------------------------- Dataset --------------------------
class DoubleSegDataset(Dataset):
    """原圖 + 線上增強版"""
    def __init__(self, img_dir, mask_dir, img_size=600):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.names = sorted([p.name for p in self.img_dir.iterdir()
                             if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        self.n = len(self.names)
        self.base_tfm = get_base_transform(img_size)
        self.aug_tfm = get_train_transform(img_size)

    def __len__(self): return self.n * 2

    def __getitem__(self, idx):
        name = self.names[idx % self.n]
        img = cv2.imread(str(self.img_dir/name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)
        mask = np.clip(mask, 0, NUM_CLASSES-1).astype(np.uint8)
        tfm = self.base_tfm if idx < self.n else self.aug_tfm
        out = tfm(image=img, mask=mask)
        return out["image"], out["mask"].long(), name

# -------------------------- Model --------------------------
try:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    DEFAULT_W = DeepLabV3_ResNet50_Weights.DEFAULT
except Exception:
    DEFAULT_W = None

def build_model(aux_loss, no_pretrain, device):
    weights = None if no_pretrain else DEFAULT_W
    ctor_aux = True if weights is not None else aux_loss
    model = deeplabv3_resnet50(weights=weights, aux_loss=ctor_aux)
    model.classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model.to(device)

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
def validate(model, loader, device, num_classes=NUM_CLASSES):
    model.eval(); conf = np.zeros((num_classes,num_classes), dtype=np.int64)
    ce = nn.CrossEntropyLoss().to(device)
    total_loss = 0.0
    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)["out"]
        total_loss += ce(logits, masks).item()
        preds = logits.argmax(1).cpu().numpy()
        for t, p in zip(masks.cpu().numpy(), preds):
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, _ = compute_miou(conf)
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)
    return miou, acc, total_loss / max(len(loader), 1)

# -------------------------- Class Weight --------------------------
def estimate_class_weights(dataset, device):
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    for _, mask, _ in loader:
        hist += np.bincount(mask.numpy().flatten(), minlength=NUM_CLASSES)
    freq = hist / max(hist.sum(), 1)
    weights = 1 / np.log(1.02 + np.maximum(freq, 1e-12))
    return torch.tensor(weights, dtype=torch.float32, device=device)

# -------------------------- Training --------------------------
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
        aux_ce = nn.CrossEntropyLoss(weight=cw)
    else:
        main_ce = nn.CrossEntropyLoss()
        aux_ce = nn.CrossEntropyLoss()

    hist_train_loss, hist_val_loss, hist_miou, hist_time = [], [], [], []
    best_miou = -1.0
    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train(); run_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            with torch.cuda.amp.autocast(True):
                out = model(imgs); logits = out["out"]
                loss = main_ce(logits, masks)
                if aux_loss and ("aux" in out):
                    loss += 0.4 * aux_ce(out["aux"], masks)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run_loss += loss.item()

        train_loss = run_loss / max(len(train_loader), 1)
        miou, acc, val_loss = validate(model, val_loader, device)
        sch.step()
        t1 = time.time()
        epoch_time = t1 - t0
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_miou.append(miou)
        hist_time.append(epoch_time)

        print(f"[{log_prefix}] Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"mIoU={miou:.4f} acc={acc:.4f} time={epoch_time:.1f}s")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), Path(save_dir)/"best.pt")
        global_best_miou = max(global_best_miou, best_miou)

    # === Plot curves ===
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(hist_train_loss, 'b-', label='Train Loss')
    ax1.plot(hist_val_loss, 'r-', label='Val Loss')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hist_miou, 'g-', label='Val mIoU')
    ax2.set_ylabel("mIoU", color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines+lines2, labels+labels2, loc="best")

    plt.title("Training & Validation Curves")
    plt.tight_layout()
    plt.savefig(Path(save_dir)/"train_curve.png", dpi=200)
    plt.close(fig)

    return best_miou

# -------------------------- Loader --------------------------
def make_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                 img_size, batch_size, seed, num_workers):
    set_seed(seed)
    train_set = DoubleSegDataset(train_img_dir, train_mask_dir, img_size)
    val_set = DoubleSegDataset(val_img_dir, val_mask_dir, img_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, batch_size//2), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# -------------------------- Optuna --------------------------
def optuna_objective(trial, base_args, device):
    args = argparse.Namespace(**vars(base_args))
    args.lr = trial.suggest_float("lr", 1e-5, 9e-5, log=True)
    args.weight_decay = trial.suggest_float("weight_decay", 1e-8, 5e-6, log=True)
    args.aux_loss = trial.suggest_categorical("aux_loss", [False, True])
    args.no_pretrain = trial.suggest_categorical("no_pretrain", [False, True])

    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size, args.seed, args.num_workers)

    best_miou = train_one_phase(
        args, device, train_loader, val_loader,
        aux_loss=args.aux_loss, no_pretrain=args.no_pretrain,
        epochs=args.proxy_epochs, save_dir=args.save_dir,
        use_class_weight=args.use_class_weight, log_prefix=f"TRIAL{trial.number:02d}")
    return best_miou

# -------------------------- Main --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_img_dir", type=str, default="UAV_dataset/train/imgs")
    p.add_argument("--train_mask_dir", type=str, default="UAV_dataset/train/masks")
    p.add_argument("--val_img_dir", type=str, default="UAV_dataset/val/imgs")
    p.add_argument("--val_mask_dir", type=str, default="UAV_dataset/val/masks")
    p.add_argument("--img_size", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--proxy_epochs", type=int, default=5)
    p.add_argument("--optuna_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0 if platform.system()=="Windows" else 4)
    p.add_argument("--use_class_weight", action="store_true", default=True)
    p.add_argument("--save_root", type=str, default="./outputs_optuna_weather")
    args = p.parse_args()

    ts = now_tag()
    args.save_dir = str(Path(args.save_root + f"_{ts}").resolve())
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Using device: {device}")

    best_json_path = Path(args.save_root) / "best_params.json"
    if best_json_path.exists():
        print(f"==> Found existing best_params.json, skip Optuna.")
        best_params = json.load(open(best_json_path, "r", encoding="utf-8"))
    else:
        print("\n==> Starting Optuna search ...")
        sampler = TPESampler(seed=args.seed, multivariate=True)
        pruner = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(lambda tr: optuna_objective(tr, args, device),
                       n_trials=args.optuna_trials, gc_after_trial=True)

        best_params = study.best_trial.params
        json.dump(best_params, open(Path(args.save_dir)/"best_params.json", "w", encoding="utf-8"), indent=2)
        print("\n==> Best params found by Optuna:\n", json.dumps(best_params, indent=2))

    print("\n==> Starting FORMAL training with best params ...")
    args.lr = best_params["lr"]
    args.weight_decay = best_params["weight_decay"]
    args.aux_loss = best_params["aux_loss"]
    args.no_pretrain = best_params["no_pretrain"]

    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size, args.seed, args.num_workers)
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
