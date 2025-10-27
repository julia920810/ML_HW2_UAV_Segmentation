# 16 類語義分割 (UAV) — 正式訓練，使用 Optuna 尋得最佳參數或手動設定。
# 自動使用已分好的 train / val 資料夾（不再自行 8:2 切分）

import os, time, json, random, argparse, platform
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

NUM_CLASSES = 16

# -------------------------- utils --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ---- Edge/Sharpen block: image-only ----
def edge_enhance_block(p=0.6):
    return A.OneOf([
        A.UnsharpMask(blur_limit=(3,7), alpha=(0.7,1.0), p=1.0),
        A.Sharpen(alpha=(0.15,0.35), lightness=(0.9,1.1), p=1.0),
        A.CLAHE(clip_limit=(2.0,4.0), tile_grid_size=(8,8), p=1.0),
        A.Emboss(alpha=(0.1,0.25), strength=(0.2,0.5), p=1.0),
    ], p=p)

# -------------------------- Dataset --------------------------
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
                        A.RandomFog(p=0.5), A.RandomRain(p=0.4),
                        A.RandomSnow(p=0.4), A.GaussNoise(p=0.4)
                    ], p=0.7),
                    A.MotionBlur(blur_limit=5, p=0.3),
                    A.ImageCompression(55, 90, p=0.35),
                    edge_enhance_block(p=0.6),
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
                        A.RandomFog(p=0.6), A.RandomRain(p=0.3),
                        A.RandomSnow(p=0.3), A.GaussNoise(p=0.3)
                    ], p=0.6),
                    A.MotionBlur(blur_limit=3, p=0.2),
                    A.ImageCompression(60, 90, p=0.3),
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
            mask = np.clip(mask, 0, NUM_CLASSES-1).astype(np.uint8)
            out = self.tfm(image=img, mask=mask)
            return out["image"], out["mask"].long(), name
        else:
            return self.tfm(image=img)["image"], name

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
def validate(model, loader, device, num_classes=NUM_CLASSES, return_conf=False):
    model.eval()
    conf = np.zeros((num_classes,num_classes), dtype=np.int64)
    val_loss = 0.0
    ce = nn.CrossEntropyLoss().to(device)
    for imgs, masks, _ in loader:
        imgs = imgs.to(device); masks = masks.to(device)
        logits = model(imgs)["out"]
        val_loss += ce(logits, masks).item()
        pred = logits.argmax(1).cpu().numpy()
        for t,p in zip(masks.cpu().numpy(), pred):
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, ious = compute_miou(conf)
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)
    val_loss /= max(len(loader), 1)
    if return_conf: return miou, acc, conf, ious, val_loss
    return miou, acc, conf, ious

# -------------------------- Model --------------------------
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

# -------------------------- Loader --------------------------
def make_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                 img_size, batch_size, seed, num_workers, augment_pack="weather"):
    set_seed(seed)
    train_set = SegDataset(train_img_dir, train_mask_dir, img_size=img_size, train=True, augment_pack=augment_pack)
    val_set   = SegDataset(val_img_dir, val_mask_dir, img_size=img_size, train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1,batch_size//2), shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# -------------------------- Main --------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_img_dir", type=str, default="UAV_dataset/dataset_split/train/imgs")
    p.add_argument("--train_mask_dir", type=str, default="UAV_dataset/dataset_split/train/masks")
    p.add_argument("--val_img_dir", type=str, default="UAV_dataset/dataset_split/val/imgs")
    p.add_argument("--val_mask_dir", type=str, default="UAV_dataset/dataset_split/val/masks")
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--aux_loss", action="store_true")
    p.add_argument("--augment_pack", type=str, default="weather", choices=["basic","weather","strong","edge"])
    p.add_argument("--save_dir", type=str, default="./outputs_final")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--params_json", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    if platform.system()=="Windows": args.num_workers=0

    # 載入 Optuna 參數
    if args.params_json and Path(args.params_json).exists():
        with open(args.params_json, "r", encoding="utf-8") as f:
            best = json.load(f)
        for k in ["lr","weight_decay","img_size","batch_size","aux_loss","no_pretrain","augment_pack"]:
            if k in best: setattr(args, k, best[k])
        print("[info] loaded best params:", best)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader = make_loaders(
        args.train_img_dir, args.train_mask_dir,
        args.val_img_dir, args.val_mask_dir,
        args.img_size, args.batch_size, args.seed, args.num_workers, args.augment_pack
    )

    # Loss
    main_ce = nn.CrossEntropyLoss(); aux_ce = nn.CrossEntropyLoss()
    model = build_model(args.aux_loss, args.no_pretrain, device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_miou=-1.0; best_path=Path(args.save_dir)/"best.pt"

    for epoch in range(1, args.epochs+1):
        model.train(); t0=time.time(); run_loss=0.0
        for imgs, masks, _ in train_loader:
            imgs=imgs.to(device); masks=masks.to(device)
            with torch.cuda.amp.autocast(True):
                out=model(imgs); logits=out["out"]
                loss=main_ce(logits,masks)
                if args.aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss+=0.4*aux_ce(out["aux"],masks)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); run_loss+=loss.item()

        miou, acc, conf, _, vloss = validate(model, val_loader, device, return_conf=True)
        sch.step()
        print(f"[Epoch {epoch:03d}] train_loss={run_loss/len(train_loader):.4f}  val_loss={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_path)
            print(f"==> New best mIoU {best_miou:.4f}, saved to {best_path}")

    print(f"==> Training done. Best mIoU = {best_miou:.4f}")

if __name__ == "__main__":
    main()
