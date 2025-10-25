# 16 類語義分割 (UAV) — 只負責進行 Optuna 超參搜尋並輸出 best_params.json
# 依賴：torch, torchvision, albumentations, optuna, opencv-python, numpy, matplotlib

import os, time, json, random, csv, argparse, platform
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(0)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.segmentation import deeplabv3_resnet50

import albumentations as A
from albumentations.pytorch import ToTensorV2

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

NUM_CLASSES = 16

# -------------------------- utils --------------------------
# ---- Edge/Sharpen block: image-only (不會作用在 mask) ----
def edge_enhance_block(p=0.8):
    """
    邊緣/對比增強（只對 image 作用；在 Albumentations 中，這些轉換不會改 mask）
    p: 套用機率
    """
    return A.OneOf([
        # 反銳化遮罩：提升局部對比感
        A.UnsharpMask(blur_limit=(3, 7), alpha=(0.7, 1.0), p=1.0),
        # 銳化（可視化邊界）
        A.Sharpen(alpha=(0.15, 0.35), lightness=(0.9, 1.1), p=1.0),
        # 局部對比增強（在霧/灰塵下讓邊界更清晰）
        A.CLAHE(clip_limit=(2.0, 4.0), tile_grid_size=(8, 8), p=1.0),
        # 浮雕（弱化版，讓邊緣更“起來”，但不至於過度）
        A.Emboss(alpha=(0.1, 0.25), strength=(0.2, 0.5), p=1.0),
    ], p=p)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=768, train=True, augment_pack="weather"):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.names = sorted([p.name for p in self.img_dir.iterdir() if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        self.train = train; self.img_size = img_size
        if train:
            if augment_pack == "basic":
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.3),

                    # ⬇️ 新增：溫和的邊緣強化
                    edge_enhance_block(p=0.35),

                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                    ToTensorV2()
                ])

            elif augment_pack == "strong":
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomResizedCrop(img_size, img_size, scale=(0.6,1.2), ratio=(0.9,1.1), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.20, rotate_limit=20,
                                    border_mode=cv2.BORDER_CONSTANT, p=0.7),
                    A.RandomBrightnessContrast(p=0.6),
                    A.ColorJitter(p=0.5),
                    A.OneOf([
                        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, alpha_coef=0.04, p=0.5),
                        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=12, blur_value=3, p=0.4),
                        A.RandomSnow(brightness_coeff=1.4, snow_point_lower=0.05, snow_point_upper=0.3, p=0.4),
                        A.GaussNoise(var_limit=(10.0,60.0), p=0.4),
                    ], p=0.7),
                    A.MotionBlur(blur_limit=5, p=0.3),
                    A.ImageCompression(quality_lower=55, quality_upper=90, p=0.35),

                    # ⬇️ 新增：強一點的邊緣強化
                    edge_enhance_block(p=0.8),

                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                    ToTensorV2()
                ])

            elif augment_pack == "edge":
                # 專注於邊界/對比，外觀干擾較少；適合你說的「不受天氣干擾也能辨識」
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    # 只做輕量的顏色/對比變化，避免整張變糊
                    A.RandomBrightnessContrast(p=0.4),
                    A.ColorJitter(p=0.25),

                    # ⬇️ 核心：邊緣/對比強化
                    edge_enhance_block(p=0.8),

                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                    ToTensorV2()
                ])

            else:  # "weather"
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(0.05, 0.10, 15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(p=0.3),
                    A.OneOf([
                        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.20, alpha_coef=0.04, p=0.6),
                        A.RandomRain(slant_lower=-8, slant_upper=8, drop_length=10, blur_value=3, p=0.3),
                        A.RandomSnow(brightness_coeff=1.4, snow_point_lower=0.05, snow_point_upper=0.25, p=0.3),
                        A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
                    ], p=0.6),
                    A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.3), p=0.3),
                    A.GlassBlur(sigma=0.2, max_delta=1, iterations=1, p=0.15),
                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.ImageCompression(60, 90, p=0.3),

                    # ⬇️ 新增：中等強度的邊緣強化
                    edge_enhance_block(p=0.5),

                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                    ToTensorV2()
                ])
        else:
            self.tfm = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]
        bgr = cv2.imread(str(self.img_dir/name), cv2.IMREAD_COLOR)
        if bgr is None: raise FileNotFoundError(f"Image not found: {self.img_dir/name}")
        img = bgr[:, :, ::-1]
        if self.mask_dir is not None:
            mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(f"Mask not found for {name}")
            mask = np.clip(mask, 0, NUM_CLASSES-1).astype(np.uint8)
            out = self.tfm(image=img, mask=mask)
            return out["image"], out["mask"].long(), name
        else:
            return self.tfm(image=img)["image"], name

def fast_hist(true, pred, num_classes):
    k = (true >= 0) & (true < num_classes)
    return np.bincount(num_classes * true[k].astype(int) + pred[k], minlength=num_classes**2).reshape(num_classes, num_classes)

def compute_miou(conf_mat):
    diag = np.diag(conf_mat).astype(np.float64)
    union = conf_mat.sum(1) + conf_mat.sum(0) - diag
    iou = diag / np.maximum(union, 1e-8)
    return np.nanmean(iou), iou

@torch.no_grad()
def validate(model, loader, device, num_classes=NUM_CLASSES, return_conf=False):
    model.eval(); conf = np.zeros((num_classes,num_classes), dtype=np.int64)
    val_loss = 0.0; ce = nn.CrossEntropyLoss().to(device)
    for imgs, masks, _ in loader:
        imgs = imgs.to(device); masks = masks.to(device)
        logits = model(imgs)["out"]
        val_loss += ce(logits, masks).item()
        pred = logits.argmax(1).cpu().numpy()
        for t,p in zip(masks.cpu().numpy(), pred):
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, _ = compute_miou(conf)
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)
    val_loss /= max(len(loader), 1)
    if return_conf: return miou, acc, conf, None, val_loss
    return miou, acc, conf, None

# 架構限制：若使用預訓練 (weights!=None) 就必須 aux_loss=True 以通過 torchvision 的檢查
try:
    from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
    DEFAULT_W = DeepLabV3_ResNet50_Weights.DEFAULT
except Exception:
    DEFAULT_W = "DEFAULT"

def build_model(aux_loss: bool, no_pretrain: bool, device):
    use_weights = None if no_pretrain else DEFAULT_W
    ctor_aux = True if use_weights is not None else aux_loss
    model = deeplabv3_resnet50(weights=use_weights, aux_loss=ctor_aux)
    model.classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[-1] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model.to(device)

# -------------------------- Optuna 目標 --------------------------

def make_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                 img_size, batch_size, seed, num_workers, augment_pack="weather"):
    """
    自動建立 train / val DataLoader，直接使用分好資料夾的資料。
    """
    set_seed(seed)

    # 建立 dataset
    train_set = SegDataset(train_img_dir, train_mask_dir,
                           img_size=img_size, train=True, augment_pack=augment_pack)
    val_set = SegDataset(val_img_dir, val_mask_dir,
                         img_size=img_size, train=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, batch_size // 2), shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader



def objective(trial, base_args):
    if platform.system()=="Windows" and base_args.num_workers!=0: base_args.num_workers=0
    # 搜尋空間
    args = argparse.Namespace(**vars(base_args))
    args.lr            = trial.suggest_float("lr", 3e-4, 3e-2, log=True)
    args.weight_decay  = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)
    args.img_size      = trial.suggest_categorical("img_size", [512, 600])
    args.batch_size    = trial.suggest_categorical("batch_size", [4, 8])
    args.aux_loss      = trial.suggest_categorical("aux_loss", [False, True])
    args.no_pretrain   = trial.suggest_categorical("no_pretrain", [False, True])
    args.augment_pack  = trial.suggest_categorical("augment_pack",["weather", "strong", "edge"])


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders(args.img_dir, args.mask_dir, args.img_size, args.batch_size, args.val_split, args.seed, args.num_workers, args.augment_pack)

    # 類別權重
    if args.use_class_weight:
        # 輕量計算：掃部分 batch 估計或直接平均像素頻率（此處簡化成均值=1，不實做重計）
        weight_t = None
        main_ce = nn.CrossEntropyLoss()
        aux_ce  = nn.CrossEntropyLoss()
    else:
        main_ce = nn.CrossEntropyLoss(); aux_ce = nn.CrossEntropyLoss()

    model = build_model(args.aux_loss, args.no_pretrain, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    T = base_args.proxy_epochs
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best = -1.0
    for epoch in range(1, T+1):
        model.train(); run_loss=0.0; t0=time.time()
        for imgs, masks, _ in train_loader:
            imgs=imgs.to(device); masks=masks.to(device)
            with torch.cuda.amp.autocast(True):
                out = model(imgs); logits = out["out"]
                loss = main_ce(logits, masks)
                if args.aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss = loss + 0.4*aux_ce(out["aux"], masks)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run_loss += loss.item()
        miou, acc, conf, _, vloss = validate(model, val_loader, device, return_conf=True)
        if sch: sch.step()
        trial.report(miou, step=epoch)
        if trial.should_prune(): raise optuna.TrialPruned()
        best = max(best, miou)
        print(f"[Epoch {epoch:03d}] train_loss={run_loss/max(len(train_loader),1):.4f}  val_loss={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}  lr={opt.param_groups[0]['lr']:.6f}  time={(time.time()-t0):.1f}s")
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, default="UAV_dataset/dataset_split/train/imgs")
    p.add_argument("--mask_dir", type=str, default="UAV_dataset/dataset_split/train/masks")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--img_size", type=int, default=512)  # 起始值，會被 trials 覆蓋
    p.add_argument("--batch_size", type=int, default=8)  # 起始值，會被 trials 覆蓋
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--aux_loss", action="store_true")
    p.add_argument("--save_dir", type=str, default="./outputs_optuna")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--optuna_trials", type=int, default=1)
    p.add_argument("--proxy_epochs", type=int, default=5)
    p.add_argument("--study_name", type=str, default="uav_seg_optuna")
    p.add_argument("--storage", type=str, default=None)
    p.add_argument("--direction", type=str, default="maximize")
    args = p.parse_args()

    set_seed(args.seed); os.makedirs(args.save_dir, exist_ok=True)
    if platform.system()=="Windows" and args.num_workers!=0:
        print("[info] Windows detected: force num_workers=0"); args.num_workers=0

    sampler = TPESampler(seed=args.seed, multivariate=True)
    pruner  = SuccessiveHalvingPruner(min_resource=3, reduction_factor=3, min_early_stopping_rate=0)
    study = optuna.create_study(direction=args.direction, sampler=sampler, pruner=pruner,
                                study_name=args.study_name, storage=args.storage, load_if_exists=bool(args.storage))

    print(f"==> Start Optuna: {args.optuna_trials} trials, proxy_epochs={args.proxy_epochs}")
    study.optimize(lambda tr: objective(tr, args), n_trials=args.optuna_trials, gc_after_trial=True)

    print("==> Best trial value (mIoU):", study.best_value)
    print("==> Best params:\n", json.dumps(study.best_trial.params, indent=2, ensure_ascii=False))

    with open(Path(args.save_dir)/"best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_trial.params, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
