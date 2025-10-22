# 16 類語義分割 (UAV) — 只負責正式訓練。可從 --params_json 載入 Optuna 輸出並可命令列覆蓋。

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

NUM_CLASSES = 16

# -------- utils (與上面保持一致) --------
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
                    A.RandomCrop(img_size, img_size), A.HorizontalFlip(p=0.5), A.ColorJitter(p=0.3),
                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
                ])
            elif augment_pack == "strong":
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomResizedCrop(img_size, img_size, scale=(0.6,1.2), ratio=(0.9,1.1), p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.20, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.7),
                    A.RandomBrightnessContrast(p=0.6), A.ColorJitter(p=0.5),
                    A.OneOf([
                        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.25, alpha_coef=0.04, p=0.5),
                        A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=12, blur_value=3, p=0.4),
                        A.RandomSnow(brightness_coeff=1.4, snow_point_lower=0.05, snow_point_upper=0.3, p=0.4),
                        A.GaussNoise(var_limit=(10.0,60.0), p=0.4),
                    ], p=0.7),
                    A.MotionBlur(blur_limit=5, p=0.3), A.ImageCompression(55, 90, p=0.35),
                    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()
                ])
            else:
                self.tfm = A.Compose([
                    A.SmallestMaxSize(max_size=img_size),
                    A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                    A.RandomCrop(img_size, img_size), A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(0.05, 0.10, 15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                    A.RandomBrightnessContrast(p=0.5), A.ColorJitter(p=0.3),
                    A.OneOf([
                        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.20, alpha_coef=0.04, p=0.6),
                        A.RandomRain(slant_lower=-8, slant_upper=8, drop_length=10, blur_value=3, p=0.3),
                        A.RandomSnow(brightness_coeff=1.4, snow_point_lower=0.05, snow_point_upper=0.25, p=0.3),
                        A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
                    ], p=0.6),
                    A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.3), p=0.3),
                    A.GlassBlur(sigma=0.2, max_delta=1, iterations=1, p=0.15),
                    A.MotionBlur(blur_limit=3, p=0.2), A.ImageCompression(60, 90, p=0.3),
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
        logits = model(imgs)["out"]; val_loss += ce(logits, masks).item()
        pred = logits.argmax(1).cpu().numpy()
        for t,p in zip(masks.cpu().numpy(), pred): conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, ious = compute_miou(conf)
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)
    val_loss /= max(len(loader), 1)
    if return_conf: return miou, acc, conf, ious, val_loss
    return miou, acc, conf, ious

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

def make_loaders(img_dir, mask_dir, img_size, batch_size, val_split, seed, num_workers, augment_pack="weather"):
    index_ds = SegDataset(img_dir, mask_dir, img_size=img_size, train=True)
    n_total = len(index_ds); n_val = max(1, int(n_total*val_split))
    g = torch.Generator().manual_seed(seed); perm = torch.randperm(n_total, generator=g)
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    train_base = SegDataset(img_dir, mask_dir, img_size=img_size, train=True, augment_pack=augment_pack)
    val_base   = SegDataset(img_dir, mask_dir, img_size=img_size, train=False)
    train_set = Subset(train_base, train_idx); val_set = Subset(val_base, val_idx)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True, persistent_workers=False)
    val_loader   = DataLoader(val_set,   batch_size=max(1,batch_size//2), shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=False)
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, default="UAV_dataset/train/imgs")
    p.add_argument("--mask_dir", type=str, default="UAV_dataset/train/masks")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--img_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--no_pretrain", action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--aux_loss", action="store_true")
    p.add_argument("--augment_pack", type=str, default="weather", choices=["basic","weather","strong"])
    p.add_argument("--save_dir", type=str, default="./outputs_final")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cm_every", type=int, default=1)
    # 讀入 Optuna 產生的最佳參數 json（可選）
    p.add_argument("--params_json", type=str, default=None, help="從 optuna_search 輸出的 best_params.json 載入，命令列可再覆蓋")
    args = p.parse_args()

    set_seed(args.seed); os.makedirs(args.save_dir, exist_ok=True)
    if platform.system()=="Windows" and args.num_workers!=0:
        print("[info] Windows detected: force num_workers=0"); args.num_workers=0

    # 若提供 best_params.json，先載入再讓命令列覆蓋
    if args.params_json is not None and Path(args.params_json).exists():
        with open(args.params_json, "r", encoding="utf-8") as f:
            best = json.load(f)
        # 承接 optuna 的鍵：lr/weight_decay/img_size/batch_size/aux_loss/no_pretrain/augment_pack
        for k in ["lr","weight_decay","img_size","batch_size","aux_loss","no_pretrain","augment_pack"]:
            if k in best: setattr(args, k, best[k])
        print("[info] loaded best params:", best)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 類別權重
    if args.use_class_weight:
        # 可依你需求實作 compute_class_weights；這裡用均值=1 的 CE 以簡化
        main_ce = nn.CrossEntropyLoss(); aux_ce = nn.CrossEntropyLoss()
    else:
        main_ce = nn.CrossEntropyLoss(); aux_ce = nn.CrossEntropyLoss()

    train_loader, val_loader = make_loaders(args.img_dir, args.mask_dir, args.img_size, args.batch_size, args.val_split, args.seed, args.num_workers, args.augment_pack)
    model = build_model(args.aux_loss, args.no_pretrain, device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    best_miou=-1.0; best_path=Path(args.save_dir)/"best.pt"
    hist = {"epoch":[],"train_loss":[],"val_loss":[],"miou":[],"acc":[],"lr":[]}

    for epoch in range(1, args.epochs+1):
        model.train(); t0=time.time(); run_loss=0.0
        for imgs, masks, _ in train_loader:
            imgs=imgs.to(device); masks=masks.to(device)
            with torch.cuda.amp.autocast(True):
                out=model(imgs); logits=out["out"]
                loss=main_ce(logits,masks)
                if args.aux_loss and ("aux" in out) and (out["aux"] is not None):
                    loss=loss+0.4*aux_ce(out["aux"],masks)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); run_loss+=loss.item()
        miou, acc, conf, ious, vloss = validate(model, val_loader, device, return_conf=True)
        if sch: sch.step()
        lr_now = opt.param_groups[0]['lr']
        avg_train = run_loss / max(len(train_loader),1)
        print(f"[Epoch {epoch:03d}] train_loss={avg_train:.4f}  val_loss={vloss:.4f}  mIoU={miou:.4f}  Acc={acc:.4f}  lr={lr_now:.6f}  time={(time.time()-t0):.1f}s")
        hist["epoch"].append(epoch); hist["train_loss"].append(avg_train); hist["val_loss"].append(vloss)
        hist["miou"].append(float(miou)); hist["acc"].append(float(acc)); hist["lr"].append(float(lr_now))
        # save last
        torch.save({"epoch":epoch,"model":model.state_dict(),"miou":float(miou),"args":vars(args)}, Path(args.save_dir)/"last.pt")
        if miou>best_miou:
            best_miou=miou; torch.save(model.state_dict(), best_path); print(f"==> New best mIoU {best_miou:.4f}, saved to {best_path}")
        # 可選：存混淆矩陣圖
        if args.cm_every>0 and (epoch % args.cm_every==0):
            plt.figure(figsize=(6,5)); plt.imshow(conf, interpolation='nearest'); plt.title('Confusion Matrix'); plt.xlabel('Pred'); plt.ylabel('True'); plt.colorbar(); plt.tight_layout(); plt.savefig(Path(args.save_dir)/f"confmat_epoch{epoch:03d}.png", dpi=150); plt.close()
    # 曲線
    try:
        plt.figure(); plt.plot(hist["epoch"], hist["train_loss"], label="train_loss"); plt.plot(hist["epoch"], hist["val_loss"], label="val_loss"); plt.legend(); plt.tight_layout(); plt.savefig(Path(args.save_dir)/"loss_curve.png", dpi=150); plt.close()
        plt.figure(); plt.plot(hist["epoch"], hist["acc"], label="acc"); plt.plot(hist["epoch"], hist["miou"], label="mIoU"); plt.legend(); plt.tight_layout(); plt.savefig(Path(args.save_dir)/"acc_miou_curve.png", dpi=150); plt.close()
    except Exception as e:
        print("[warn] plotting failed:", e)

    with open(Path(args.save_dir)/"run_log.txt", "w", encoding="utf-8") as f:
        f.write("=== Training Run Log ===\n")
        f.write(time.strftime("Date: %Y-%m-%d %H:%M:%S\n"))
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")
        f.write(f"Best mIoU: {best_miou:.6f}\n")

if __name__ == "__main__":
    main()
