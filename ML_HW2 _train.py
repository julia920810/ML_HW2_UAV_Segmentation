# train_seg16_simple.py
# ------------------------------------------------------------
# 16 類語義分割 (像素值 0..15)，UAV 空拍；PyTorch + Torchvision
# 精簡版重點：
# 1) Dataset + Augment（含天氣域轉移增強，已改用 A.ImageCompression）
# 2) DeepLabV3(ResNet50) + CrossEntropy(+可選 class weight)
# 3) 驗證: mIoU / overall accuracy
# 4) 儲存最佳模型 best.pt
# ------------------------------------------------------------

import os, time, json, random                   # 作業系統、計時、JSON、隨機數
from pathlib import Path                        # 路徑物件
from typing import Tuple, List                  # 型別標註（可讀性）
import argparse                                 # 解析命令列參數
import numpy as np                              # 數值運算
import cv2                                      # 影像讀寫（OpenCV）

import torch                                    # PyTorch 主套件
import torch.nn as nn                           # 神經網路模組
import torch.nn.functional as F                 # 函數類別（未於此檔顯性使用）
from torch.utils.data import Dataset, DataLoader, random_split  # 資料集/資料載入/隨機切分
from torchvision.models.segmentation import deeplabv3_resnet50  # 分割模型

import albumentations as A                      # 影像前處理/增強
from albumentations.pytorch import ToTensorV2   # 轉成 PyTorch Tensor 的轉換器

# --------------------------
# 工具：設定隨機種子（可重現性）
# --------------------------
def set_seed(seed=42):
    random.seed(seed)                           # Python 標準隨機
    np.random.seed(seed)                        # NumPy 隨機
    torch.manual_seed(seed)                     # PyTorch CPU 隨機
    torch.cuda.manual_seed_all(seed)            # PyTorch GPU 隨機（多 GPU）
    torch.backends.cudnn.deterministic = False  # 不鎖定 cuDNN 演算法（較快，但非完全可重現）
    torch.backends.cudnn.benchmark = True       # 讓 cuDNN 自動尋找最佳卷積實作（加速）

# --------------------------
# Dataset
# --------------------------
class SegDataset(Dataset):
    """
    img_dir: 影像資料夾
    mask_dir: 對應遮罩資料夾（灰階 PNG，像素值 0..15）
    train=True 時使用較強增強；False 時用測試/驗證前處理
    """
    def __init__(self, img_dir, mask_dir=None, img_size=768, train=True):
        self.img_dir = Path(img_dir)            # 影像根目錄
        self.mask_dir = Path(mask_dir) if mask_dir else None  # 標註根目錄（可能為 None）
        # 收集影像檔名（副檔名限 png/jpg/jpeg）
        self.names = sorted([p.name for p in self.img_dir.iterdir()
                             if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        self.train = train                      # 是否為訓練模式
        self.img_size = img_size                # 目標輸入尺寸（方形）

        if train:
            # ======== 訓練前處理 + 資料增強 ========
            self.tfm = A.Compose([
                A.SmallestMaxSize(max_size=img_size),  # 縮放，確保短邊至少到 img_size
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomCrop(width=img_size, height=img_size),  # 不會再比圖大
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
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
                A.ImageCompression(quality_lower=60, quality_upper=90, p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            # ======== 驗證/測試前處理（不做隨機增強） ========
            self.tfm = A.Compose([
                A.LongestMaxSize(max_size=img_size),                       # 等比縮放：最長邊 <= img_size
                A.PadIfNeeded(                                            # 若非方形則補邊到 (img_size, img_size)
                    min_height=img_size, min_width=img_size,
                    border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0
                ),
                A.Normalize(                                               # 同訓練的 Normalize
                    mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)
                ),
                ToTensorV2()                                               # 轉 Tensor
            ])

    def __len__(self): 
        return len(self.names)                                             # 回傳影像數量

    def __getitem__(self, i):
        name = self.names[i]                                               # 第 i 個檔名
        # 讀入彩色影像（BGR），並轉成 RGB（模型常用 RGB）
        img = cv2.imread(str(self.img_dir/name), cv2.IMREAD_COLOR)[:, :, ::-1]
        if self.mask_dir is not None:
            # 讀入灰階遮罩（標籤圖；每個像素是類別 id 0..15）
            mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found for {name}")
            mask = np.clip(mask, 0, 15).astype(np.uint8)                   # 保障標籤落在 0..15
            out = self.tfm(image=img, mask=mask)                           # 同步對影像/遮罩做增強與前處理
            return out["image"], out["mask"].long(), name                  # 回傳 Tensor 影像、長整數遮罩、檔名
        else:
            return self.tfm(image=img)["image"], name                      # 無標註模式僅回傳影像與檔名

# --------------------------
# 混淆矩陣 / mIoU
# --------------------------
def fast_hist(true, pred, num_classes):
    k = (true >= 0) & (true < num_classes)                                 # 合法標籤遮罩
    # 以 bincount 快速累積混淆矩陣（row=true, col=pred）
    return np.bincount(num_classes * true[k].astype(int) + pred[k],
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)

def compute_miou(conf_mat):
    diag = np.diag(conf_mat).astype(np.float64)                             # 各類別 TP
    union = conf_mat.sum(1) + conf_mat.sum(0) - diag                        # 各類別聯集 = (row sum + col sum - TP)
    iou = diag / np.maximum(union, 1e-8)                                    # IoU = TP / Union
    miou = np.nanmean(iou)                                                  # 16 類別平均
    return miou, iou

# --------------------------
# 類別權重（可緩解不平衡）
# --------------------------
def compute_class_weights(mask_dir: str, num_classes=16):
    hist = np.zeros(num_classes, dtype=np.int64)                            # 累積每類像素數
    mask_dir = Path(mask_dir)
    names = [p.name for p in mask_dir.iterdir() if p.suffix.lower() == ".png"]  # 只掃 png
    for n in names:
        m = cv2.imread(str(mask_dir/n), cv2.IMREAD_GRAYSCALE)               # 讀遮罩
        if m is None: continue
        for c in range(num_classes):
            hist[c] += np.sum(m == c)                                       # 計算每類像素次數
    freq = hist / max(hist.sum(), 1)                                        # 類別相對頻率
    inv = 1.0 / np.clip(freq, 1e-6, None)                                   # 反比例（稀有類較大權重）
    class_weight = (inv / inv.mean()).astype(np.float32)                    # 正規化到均值=1
    return freq, class_weight

# --------------------------
# 驗證流程
# --------------------------
@torch.no_grad()                                                            # 驗證不需要梯度
def validate(model, loader, device, num_classes=16):
    model.eval()                                                            # 切換 eval（關掉 Dropout/BN 更新）
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)             # 初始化混淆矩陣
    for imgs, masks, _ in loader:                                           # 逐 batch
        imgs = imgs.to(device)                                              # 影像 -> 裝置（GPU/CPU）
        masks_np = masks.numpy()                                            # GT 轉 numpy（CPU 計算混淆矩陣）
        logits = model(imgs)["out"]                                         # 前向輸出（N,C,H,W）
        pred = logits.argmax(1).cpu().numpy()                               # 取 argmax 得類別預測（N,H,W）
        for t, p in zip(masks_np, pred):                                    # 逐張累積混淆矩陣
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)
    miou, ious = compute_miou(conf)                                         # mIoU 與 per-class IoU
    acc = np.diag(conf).sum() / conf.sum().clip(min=1)                      # Overall Accuracy
    return miou, acc, conf, ious

# --------------------------
# 主程式：訓練/驗證
# --------------------------
def main():
    parser = argparse.ArgumentParser()                                      # 建立參數解析器
    parser.add_argument("--img_dir", type=str, default="UAV_dataset/train/imgs")     # 訓練影像資料夾
    parser.add_argument("--mask_dir", type=str, default="UAV_dataset/train/masks")   # 訓練遮罩資料夾
    parser.add_argument("--val_split", type=float, default=0.1)             # 驗證比例
    parser.add_argument("--img_size", type=int, default=768)                # 輸入尺寸（邊長）
    parser.add_argument("--batch_size", type=int, default=8)                # 批次大小
    parser.add_argument("--epochs", type=int, default=60)                   # 訓練 epoch 數
    parser.add_argument("--lr", type=float, default=3e-4)                   # 學習率
    parser.add_argument("--weight_decay", type=float, default=1e-4)         # 權重衰退（L2）
    parser.add_argument("--no_pretrain", action="store_true")               # 不載 ImageNet 預訓練權重
    parser.add_argument("--use_class_weight", action="store_true",          # 是否使用 class weight
                        help="開啟後 CrossEntropy 會帶入由資料計算的 class weight")
    parser.add_argument("--save_dir", type=str, default="./outputs_simple") # 輸出資料夾
    parser.add_argument("--seed", type=int, default=42)                     # 隨機種子
    parser.add_argument("--num_workers", type=int, default=4)               # DataLoader 工人數
    args = parser.parse_args()                                              # 解析參數

    set_seed(args.seed)                                                     # 固定隨機性
    os.makedirs(args.save_dir, exist_ok=True)                               # 建立輸出資料夾

    # 類別權重（可選）
    if args.use_class_weight:
        print("==> Compute class freq/weights ...")
        freq, class_weight = compute_class_weights(args.mask_dir, num_classes=16)  # 統計類別像素分佈
        print("freq:", np.round(freq, 6))                                   # 印出類別比例
        print("class_weight:", np.round(class_weight, 3))                   # 印出類別權重
    else:
        class_weight = None                                                 # 不使用 class weight

    # 建立 Dataset + 切分
    full = SegDataset(args.img_dir, args.mask_dir, img_size=args.img_size, train=True)  # 訓練模式（含增強）
    n_total = len(full)                                                     # 全部張數
    n_val = max(1, int(n_total * args.val_split))                           # 驗證數量（最少 1）
    n_train = n_total - n_val                                               # 訓練數量
    train_set, val_set = random_split(                                      # 隨機切分
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)                  # 固定切分
    )
    val_set.dataset.train = False                                           # 把驗證那份資料的 transform 切到 eval 前處理

    # 建立 DataLoader（訓練隨機打亂；驗證不打亂）
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size // 2), shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 選 GPU/CPU
    print("==> Device:", device)

    # 建立模型（ResNet50 編碼器的 DeepLabV3）
    model = deeplabv3_resnet50(weights=None if args.no_pretrain else "DEFAULT")  # 是否載入 ImageNet 預訓練
    model.classifier[-1] = nn.Conv2d(256, 16, kernel_size=1)               # 替換分類頭輸出通道為 16 類
    model = model.to(device)                                               # 模型移到裝置

    # Loss（可選 class weight）
    if class_weight is not None:
        weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)  # 權重張量
        ce_loss = nn.CrossEntropyLoss(weight=weight_t)                             # 帶權重的 CE
    else:
        ce_loss = nn.CrossEntropyLoss()                                            # 一般 CE

    # Optimizer + Scheduler（用 AdamW + 餘弦退火）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=True)                     # 啟用自動混合精度（AMP）加速

    best_miou = -1.0                                                     # 紀錄最佳 mIoU
    best_path = Path(args.save_dir) / "best.pt"                          # 最佳模型路徑

    for epoch in range(1, args.epochs + 1):                              # 逐 epoch 訓練
        model.train()                                                    # 訓練模式
        t0 = time.time()                                                 # 計時
        run_loss = 0.0                                                   # 累積訓練損失

        for it, (imgs, masks, _) in enumerate(train_loader, 1):          # 逐 batch
            imgs = imgs.to(device); masks = masks.to(device)             # 資料移到裝置
            with torch.cuda.amp.autocast(enabled=True):                  # 混合精度訓練
                logits = model(imgs)["out"]                              # 前向輸出（N,C,H,W）
                loss = ce_loss(logits, masks)                            # 交叉熵損失

            optimizer.zero_grad(set_to_none=True)                        # 梯度清空（更快）
            scaler.scale(loss).backward()                                # 反向傳播（AMP）
            scaler.step(optimizer)                                       # 更新參數（AMP）
            scaler.update()                                              # 更新 AMP 的縮放因子
            run_loss += loss.item()                                      # 累加損失

        # ---- 驗證 ----
        miou, acc, conf, ious = validate(model, val_loader, device)      # 計算 mIoU/Acc
        if scheduler: scheduler.step()                                    # 更新學習率

        print(f"[Epoch {epoch:03d}] "
              f"loss={run_loss/len(train_loader):.4f}  "
              f"mIoU={miou:.4f}  Acc={acc:.4f}  "
              f"time={(time.time()-t0):.1f}s")

        # 每個 epoch 存「last」
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "miou": float(miou),
            "args": vars(args)
        }, Path(args.save_dir) / "last.pt")

        # 若此 epoch 的 mIoU 更佳，覆寫 best.pt
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), best_path)
            print(f"==> New best mIoU {best_miou:.4f}, saved to {best_path}")

    # 訓練完成：寫 run_log.txt（簡版）
    with open(Path(args.save_dir) / "run_log.txt", "w", encoding="utf-8") as f:
        f.write("=== Training Run Log ===\n")
        f.write(time.strftime("Date: %Y-%m-%d %H:%M:%S\n"))
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")
        f.write(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")
        f.write(f"Best mIoU: {best_miou:.6f}\n")

if __name__ == "__main__":
    main()                                                               # 進入主程式
