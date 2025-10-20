# train_seg16.py
# ------------------------------------------------------------
# 16 類語義分割 (像素值 0..15)，UAV 空拍場景；PyTorch + Torchvision
# 特色：
# 1) 資料健檢：自動計算類別像素分佈與 class weight（緩解不平衡）
# 2) 訓練：DeepLabV3(ResNet50/101) + CE(可加權) + Dice 混合損失
# 3) 驗證：mIoU / overall accuracy / per-class IoU
# 4) 推論：滑窗拼接 + TTA(Flip + 多尺度)；輸出灰階 0..15 PNG
# 5) (可選) 輸出每類 RLE CSV（提交用）
# ------------------------------------------------------------

import os, sys, math, time, argparse, json, random
from pathlib import Path
import numpy as np
import cv2
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use("Agg")  # 後端不顯示，直接存檔
import matplotlib.pyplot as plt
import pandas as pd
# --------------------------

# 工具：設定隨機種子，讓結果更可重現
# --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# --------------------------
# Dataset
# --------------------------
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, img_size=768, train=True):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.names = sorted([p.name for p in self.img_dir.iterdir()
                             if p.suffix.lower() in [".png",".jpg",".jpeg"]])
        self.train = train
        self.img_size = img_size

        if train:
            self.tfm = A.Compose([
                A.RandomCrop(width=img_size, height=img_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.3),
                A.OneOf([
                    A.RandomFog(fog_coef_lower=0.06, fog_coef_upper=0.2, alpha_coef=0.04, p=0.5),
                    A.RandomRain(slant_lower=-8, slant_upper=8, drop_length=10, blur_value=3, p=0.3),
                    A.RandomSnow(brightness_coeff=1.5, snow_point_lower=0.1, snow_point_upper=0.3, p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                ], p=0.5),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.JpegCompression(quality_lower=60, quality_upper=90, p=0.3),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.tfm = A.Compose([
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self): return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = cv2.imread(str(self.img_dir/name), cv2.IMREAD_COLOR)[:,:,::-1]
        if self.mask_dir is not None:
            mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)
            if mask is None: raise FileNotFoundError(f"Mask not found for {name}")
            mask = np.clip(mask, 0, 15).astype(np.uint8)
            out = self.tfm(image=img, mask=mask)
            return out["image"], out["mask"].long(), name
        else:
            return self.tfm(image=img)["image"], name

# --------------------------
# Dice Loss
# --------------------------
class DiceLoss(nn.Module):
    def __init__(self, num_classes=16, smooth=1.0, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        C = probs.shape[1]
        loss = 0.0
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (targets != self.ignore_index)

        for c in range(C):
            pc = probs[:, c, :, :]
            tc = (targets == c).float()
            if valid_mask is not None:
                pc = pc * valid_mask
                tc = tc * valid_mask
            inter = (pc * tc).sum(dim=(1,2))
            denom = pc.sum(dim=(1,2)) + tc.sum(dim=(1,2))
            dice = (2.0*inter + self.smooth) / (denom + self.smooth)
            loss += (1.0 - dice).mean()
        return loss / C

# --------------------------
# metrics / utils
# --------------------------
def fast_hist(true, pred, num_classes):
    k = (true >= 0) & (true < num_classes)
    return np.bincount(num_classes * true[k].astype(int) + pred[k],
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)

def compute_miou(conf_mat):
    diag = np.diag(conf_mat).astype(np.float64)
    union = conf_mat.sum(1) + conf_mat.sum(0) - diag
    iou = diag / np.maximum(union, 1e-8)
    miou = np.nanmean(iou)
    return miou, iou

def rle_encode(bin_mask: np.ndarray) -> str:
    pixels = bin_mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[:-1:2]
    return ' '.join(map(str, runs))

# --------------------------
# 視覺化：混淆矩陣 / 曲線 / 範例 overlay
# --------------------------
def save_confusion(conf, save_path):
    plt.figure(figsize=(7,6))
    cm = conf / np.maximum(conf.sum(axis=1, keepdims=True), 1)
    plt.imshow(cm, interpolation='nearest')
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_curves(history, save_dir):
    # history keys: train_total, train_ce, train_dice, val_miou, val_acc
    x = np.arange(1, len(history["train_total"])+1)
    # Loss
    plt.figure()
    plt.plot(x, history["train_total"], label="Train Total Loss")
    plt.plot(x, history["train_ce"], label="Train CE")
    plt.plot(x, history["train_dice"], label="Train Dice")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss Curves")
    plt.tight_layout(); plt.savefig(Path(save_dir)/"curves_loss.png", dpi=150); plt.close()
    # Acc, mIoU
    plt.figure()
    plt.plot(x, history["val_acc"], label="Val Acc")
    plt.plot(x, history["val_miou"], label="Val mIoU")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Validation Curves")
    plt.tight_layout(); plt.savefig(Path(save_dir)/"curves_metrics.png", dpi=150); plt.close()

def colorize_mask(mask):
    # 16 色簡易調色盤（可自行調整）
    palette = np.array([
        [  0,  0,  0],[128, 64,128],[244, 35,232],[ 70, 70, 70],
        [102,102,156],[190,153,153],[153,153,153],[250,170, 30],
        [220,220,  0],[107,142, 35],[152,251,152],[ 70,130,180],
        [220, 20, 60],[255,  0,  0],[  0,  0,142],[  0, 60,100]
    ], dtype=np.uint8)
    h, w = mask.shape
    color = np.zeros((h,w,3), dtype=np.uint8)
    for cid in range(16):
        color[mask==cid] = palette[cid % len(palette)]
    return color

def save_overlay(rgb, pred, save_path, alpha=0.5):
    color = colorize_mask(pred)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(rgb, 1-alpha, color, alpha, 0)
    cv2.imwrite(str(save_path), overlay)

# --------------------------
# 推論：滑窗 + TTA
# --------------------------
@torch.no_grad()
def sliding_window_tta(model, img_rgb, device, window=1024, overlap=256,
                       scales=(1.0, 1.25, 0.75), do_flip=True, num_classes=16):
    H, W, _ = img_rgb.shape
    stride = window - overlap
    yy = list(range(0, max(H-window, 0)+1, stride))
    xx = list(range(0, max(W-window, 0)+1, stride))
    if yy[-1] + window < H: yy.append(H - window)
    if xx[-1] + window < W: xx.append(W - window)

    wy = cv2.getGaussianKernel(window, window/6).astype(np.float32)
    wx = cv2.getGaussianKernel(window, window/6).astype(np.float32)
    win2d = (wy @ wx.T); win2d = win2d / win2d.max()

    prob_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    def prep(t_img):
        aug = A.Compose([
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2()
        ])
        out = aug(image=t_img)
        return out["image"].unsqueeze(0).to(device)

    for sy in yy:
        for sx in xx:
            tile = img_rgb[sy:sy+window, sx:sx+window, :]
            tile_probs_accum = torch.zeros((num_classes, window, window), device=device)
            for sc in scales:
                t_res = cv2.resize(tile, (int(window*sc), int(window*sc)), interpolation=cv2.INTER_LINEAR) if sc!=1.0 else tile
                x0 = prep(t_res)
                flips = [lambda x: x] if not do_flip else [lambda x: x, lambda x: x.flip(-1)]
                for f in flips:
                    x = f(x0)
                    logits = model(x)["out"]
                    logits = F.interpolate(logits, size=(window, window), mode="bilinear", align_corners=False)
                    probs = F.softmax(logits, dim=1)[0]
                    if do_flip and f is not None:
                        probs = probs.flip(-1)
                    tile_probs_accum += probs
            tile_probs_accum /= (len(scales) * (2 if do_flip else 1))
            tile_probs_np = tile_probs_accum.detach().float().cpu().numpy()
            prob_sum[:, sy:sy+window, sx:sx+window] += tile_probs_np * win2d[None, :, :]
            weight_sum[sy:sy+window, sx:sx+window] += win2d
    prob_sum /= np.maximum(weight_sum[None, :, :], 1e-8)
    return prob_sum.argmax(axis=0).astype(np.uint8)

# --------------------------
# 類別權重
# --------------------------
def compute_class_weights(mask_dir: str, num_classes=16):
    hist = np.zeros(num_classes, dtype=np.int64)
    mask_dir = Path(mask_dir)
    names = [p.name for p in mask_dir.iterdir() if p.suffix.lower()==".png"]
    for n in names:
        m = cv2.imread(str(mask_dir/n), cv2.IMREAD_GRAYSCALE)
        if m is None: continue
        for c in range(num_classes):
            hist[c] += np.sum(m==c)
    freq = hist / max(hist.sum(), 1)
    inv = 1.0 / np.clip(freq, 1e-6, None)
    class_weight = (inv / inv.mean()).astype(np.float32)
    return freq, class_weight

# --------------------------
# 訓練/驗證
# --------------------------
def train_one_epoch(model, loader, optimizer, scaler, ce_loss, dice_loss,
                    device, epoch, log_interval=50, lambda_dice=0.5):
    model.train()
    m = {"ce":0.0,"dice":0.0,"total":0.0}
    t0 = time.time()
    for it,(imgs, masks, _) in enumerate(loader,1):
        imgs = imgs.to(device); masks = masks.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            out = model(imgs)["out"]
            loss_ce = ce_loss(out, masks)
            loss_dice = dice_loss(out, masks)
            loss = loss_ce + lambda_dice * loss_dice
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        m["ce"] += loss_ce.item(); m["dice"] += loss_dice.item(); m["total"] += loss.item()
        if it % log_interval == 0:
            print(f"[Epoch {epoch}] it {it}/{len(loader)} CE={m['ce']/it:.4f} Dice={m['dice']/it:.4f} Total={m['total']/it:.4f}")
    return {k:v/len(loader) for k,v in m.items()}

@torch.no_grad()
def validate(model, loader, device, num_classes=16):
    model.eval()
    conf = np.zeros((num_classes,num_classes), dtype=np.int64)
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        masks_np = masks.numpy()
        logits = model(imgs)["out"]
        pred = logits.argmax(1).cpu().numpy()
        for t, p in zip(masks_np, pred):
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, ious = compute_miou(conf)
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)
    return miou, acc, conf, ious

# --------------------------
# 模型建立：可插入 Dropout2d（分類頭前）
# --------------------------
class HeadWithDropout(nn.Module):
    def __init__(self, in_ch=256, num_classes=16, p=0.0):
        super().__init__()
        self.drop = nn.Dropout2d(p) if p>0 else nn.Identity()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)
    def forward(self, x):
        return self.conv(self.drop(x))

def build_model(model_name="deeplabv3_resnet50", num_classes=16, pretrained=True, head_dropout=0.0):
    if model_name == "deeplabv3_resnet50":
        model = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
    elif model_name == "deeplabv3_resnet101":
        model = deeplabv3_resnet101(weights="DEFAULT" if pretrained else None)
    else:
        raise ValueError("model_name must be deeplabv3_resnet50 or deeplabv3_resnet101")
    # 替換分類頭，插入 Dropout2d
    model.classifier[-1] = HeadWithDropout(256, num_classes, p=head_dropout)
    return model

# --------------------------
# 寫 run_log.txt（紀錄所有實驗資訊）
# --------------------------
def write_run_log(args, save_dir, model, class_freq, class_weight, best_miou):
    log_path = Path(save_dir) / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Training Run Log ===\n")
        f.write(time.strftime("Date: %Y-%m-%d %H:%M:%S\n"))
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n\n")
        # 超參
        f.write("[Hyperparameters]\n")
        f.write(f"model={args.model}\n")
        f.write(f"img_size={args.img_size}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"epochs={args.epochs}\n")
        f.write(f"optimizer={args.optimizer}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"weight_decay={args.weight_decay}\n")
        f.write(f"scheduler={args.scheduler}\n")
        f.write(f"lambda_dice={args.lambda_dice}\n")
        f.write(f"head_dropout={args.head_dropout}\n")
        f.write(f"activation=softmax (for eval)\n")
        f.write(f"loss=CrossEntropy(weighted)+Dice\n")
        f.write(f"img_dir={args.img_dir}\nmask_dir={args.mask_dir}\nval_split={args.val_split}\n")
        f.write("\n[Class Frequency]\n")
        f.write(json.dumps(class_freq.tolist(), indent=2))
        f.write("\n[Class Weight]\n")
        f.write(json.dumps(class_weight.tolist(), indent=2))
        f.write("\n\n[Model Summary]\n")
        f.write(str(model))
        f.write(f"\n\nBest mIoU: {best_miou:.6f}\n")
        f.write("\nArtifacts:\n")
        f.write("- curves_loss.png\n- curves_metrics.png\n- confusion_matrix.png\n")
        f.write("- sample_overlays/*.png (if generated)\n")
        if args.test_dir:
            f.write("- preds/*.png (if --save_pred_png)\n- sample_submission.csv (if --gen_rle_csv)\n")


# --------------------------
# 主流程：解析參數、建資料集/模型、訓練、驗證、推論
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    # ====== 訓練相關 ======
    parser.add_argument("--img_dir", type=str, default="UAV_dataset/train/imgs", help="訓練影像資料夾 (e.g., ./train/imgs)")
    parser.add_argument("--mask_dir", type=str,default="UAV_dataset/train/masks", help="訓練標註資料夾 (e.g., ./train/masks)")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="deeplabv3_resnet50",
                        choices=["deeplabv3_resnet50","deeplabv3_resnet101"])
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw","sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine","none"])
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--head_dropout", type=float, default=0.0, help="分類頭 Dropout2d 機率")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda_dice", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)

    # ====== 推論/提交相關 ======
    parser.add_argument("--test_dir", type=str, default="UAV_dataset/test", help="測試影像資料夾 (無標)")
    parser.add_argument("--infer_window", type=int, default=1024)
    parser.add_argument("--infer_overlap", type=int, default=256)
    parser.add_argument("--infer_scales", type=str, default="1.0,1.25,0.75")
    parser.add_argument("--infer_flip", action="store_true")
    parser.add_argument("--save_pred_png", action="store_true")
    parser.add_argument("--pred_dir", type=str, default="./outputs/preds")
    parser.add_argument("--gen_rle_csv", action="store_true")
    parser.add_argument("--rle_csv", type=str, default="./outputs/sample_submission.csv")

    args = parser.parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

   # 額外：輸出可視化樣本數
    parser.add_argument("--viz_samples", type=int, default=8, help="從驗證集中存幾張 overlay 畫面")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(Path(args.save_dir)/"sample_overlays", exist_ok=True)

    # 類別權重
    print("==> Compute class freq/weights ...")
    freq, class_weight = compute_class_weights(args.mask_dir, num_classes=16)
    print("freq:", np.round(freq,6))
    print("class_weight:", np.round(class_weight,3))

    # 資料集/切分
    full = SegDataset(args.img_dir, args.mask_dir, img_size=args.img_size, train=True)
    n_total = len(full); n_val = int(n_total*args.val_split); n_train = n_total - n_val
    train_set, val_set = random_split(full, [n_train, n_val],
                                      generator=torch.Generator().manual_seed(args.seed))
    val_set.dataset.train = False

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1,args.batch_size//2), shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==> Device:", device)

    model = build_model(args.model, num_classes=16,
                        pretrained=(not args.no_pretrain),
                        head_dropout=args.head_dropout).to(device)

    # 優化器/排程
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay, nesterov=True)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)
    ce_loss = nn.CrossEntropyLoss(weight=weight_t)
    dice_loss = DiceLoss(num_classes=16)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    history = {"train_total":[], "train_ce":[], "train_dice":[], "val_miou":[], "val_acc":[]}
    best_miou = -1.0
    best_path = Path(args.save_dir)/"best.pt"

    for epoch in range(1, args.epochs+1):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, ce_loss, dice_loss,
                             device, epoch, lambda_dice=args.lambda_dice)
        miou, acc, conf, ious = validate(model, val_loader, device)
        if scheduler: scheduler.step()

        history["train_total"].append(tr["total"])
        history["train_ce"].append(tr["ce"])
        history["train_dice"].append(tr["dice"])
        history["val_miou"].append(float(miou))
        history["val_acc"].append(float(acc))

        print(f"[Val {epoch}] mIoU={miou:.4f} Acc={acc:.4f} | IoU={np.round(ious,3)}")

        # 每個 epoch 存 last
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "miou": miou,
            "args": vars(args)
        }, Path(args.save_dir)/"last.pt")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_path)
            print(f"==> New best mIoU {best_miou:.4f}, saved to {best_path}")

    # 繪圖保存
    save_curves(history, args.save_dir)
    save_confusion(conf, Path(args.save_dir)/"confusion_matrix.png")

    # 取幾張驗證集樣本做 overlay 可視化
    if args.viz_samples > 0 and len(val_set) > 0:
        model.eval()
        cnt = 0
        for imgs, masks, names in DataLoader(val_set, batch_size=1, shuffle=True):
            with torch.no_grad():
                logits = model(imgs.to(device))["out"]
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            # 還原到 numpy 圖供 overlay
            # 這裡 val_set.dataset.train=False，前處理包含 Normalize，因此只做可視化用原始讀檔：
            raw = cv2.imread(str(Path(args.img_dir)/names[0]), cv2.IMREAD_COLOR)[:,:,::-1]
            save_overlay(raw, pred, Path(args.save_dir)/"sample_overlays"/f"{names[0]}")
            cnt += 1
            if cnt >= args.viz_samples: break

    # 訓練完成後寫日誌
    write_run_log(args, args.save_dir, model, freq, class_weight, best_miou)

    # ----------------------
    # 推論 / Kaggle CSV
    # ----------------------
    if args.test_dir is not None:
        print("==> Inference on test ...")
        pred_dir = Path(args.pred_dir); pred_dir.mkdir(parents=True, exist_ok=True)
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state); model.eval()

        scales = tuple(float(s) for s in args.infer_scales.split(",") if s.strip())
        test_names = sorted([p.name for p in Path(args.test_dir).iterdir()
                             if p.suffix.lower() in [".png",".jpg",".jpeg"]])

        if args.gen_rle_csv:
            import csv
            csv_fp = open(args.rle_csv, "w", newline="")
            writer = csv.writer(csv_fp)
            header = ["img"] + [f"class_{i}" for i in range(16)]
            writer.writerow(header)

        for i, name in enumerate(test_names, 1):
            img = cv2.imread(str(Path(args.test_dir)/name), cv2.IMREAD_COLOR)[:,:,::-1]
            pred = sliding_window_tta(model, img, device,
                                      window=args.infer_window, overlap=args.infer_overlap,
                                      scales=scales, do_flip=args.infer_flip, num_classes=16)
            if args.save_pred_png:
                cv2.imwrite(str(pred_dir/(Path(name).stem+".png")), pred)
            if args.gen_rle_csv:
                row = [name]
                for cid in range(16):
                    binmask = (pred == cid).astype(np.uint8)
                    row.append("none" if binmask.sum()==0 else rle_encode(binmask))
                writer.writerow(row)
            if i % 20 == 0:
                print(f"Infer {i}/{len(test_names)} ...")

        if args.gen_rle_csv:
            csv_fp.close()
            print(f"==> Kaggle CSV saved: {args.rle_csv}")
        print("==> Inference done.")

if __name__ == "__main__":
    main()