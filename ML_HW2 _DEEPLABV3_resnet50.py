# train_seg16_simple.py
# ------------------------------------------------------------
# 16 類語義分割 (像素值 0..15)，UAV 空拍；PyTorch + Torchvision
# 重點：
# 1) 正確 train/val 切分（獨立 Dataset 實例 + 同索引）
# 2) 可切換 DeepLabV3 aux head（預設關閉避免推論不相容）
# 3) 驗證: mIoU / overall accuracy（含混淆矩陣）
# 4) 自動輸出 logs 與曲線圖（loss / acc / mIoU）
# 5) 儲存最佳模型 best.pt 與 last.pt
# ------------------------------------------------------------

import os, time, json, random, csv                           # 常用工具：路徑/時間/JSON/隨機/CSV
from pathlib import Path                                     # 路徑物件（更安全易讀）
from typing import Tuple, List                               # 型別標註（可讀性）
import argparse                                              # 解析命令列參數
import numpy as np                                           # 數值計算
import cv2                                                   # 影像讀寫（OpenCV）
cv2.setNumThreads(0)
import matplotlib                                            # 畫圖後端設定
matplotlib.use("Agg")                                        # 使用無視窗後端，能在伺服器儲存圖
import matplotlib.pyplot as plt                              # 畫圖
import platform
import torch                                                 # PyTorch 主套件
import torch.nn as nn                                        # 神經網路模組
from torch.utils.data import Dataset, DataLoader, Subset     # 資料集/資料載入/子集
from torchvision.models.segmentation import deeplabv3_resnet50  # DeepLabV3 模型

import albumentations as A                                   # 影像增強/前處理
from albumentations.pytorch import ToTensorV2                # 轉成 PyTorch Tensor

# --------------------------
# 工具：設定隨機種子（可重現性）
# --------------------------
def set_seed(seed=42):                                       # 設定隨機種子以利重現
    random.seed(seed)                                        # Python 內建隨機
    np.random.seed(seed)                                     # NumPy 隨機
    torch.manual_seed(seed)                                  # PyTorch CPU 隨機
    torch.cuda.manual_seed_all(seed)                         # PyTorch GPU 隨機（多 GPU）
    torch.backends.cudnn.deterministic = False               # 不鎖定 cuDNN（速度較快）
    torch.backends.cudnn.benchmark = True                    # 讓 cuDNN 自動挑最佳卷積實作

# --------------------------
# Dataset
# --------------------------
class SegDataset(Dataset):                                   # 自訂語義分割資料集
    """
    img_dir: 影像資料夾
    mask_dir: 對應遮罩資料夾（灰階 PNG，像素值 0..15）
    train=True 時使用增強；False 時用驗證/測試前處理
    """
    def __init__(self, img_dir, mask_dir=None, img_size=768, train=True):  # 建構子
        self.img_dir = Path(img_dir)                                       # 影像根目錄
        self.mask_dir = Path(mask_dir) if mask_dir else None               # 標註根目錄（可為 None）
        self.names = sorted([p.name for p in self.img_dir.iterdir()        # 收集檔名（只取圖檔）
                             if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        self.train = train                                                 # 是否訓練模式（決定 transform）
        self.img_size = img_size                                           # 目標邊長（方形）

        if train:                                                          # 訓練模式：使用隨機增強
            self.tfm = A.Compose([
                A.SmallestMaxSize(max_size=img_size),                      # 短邊放大到 >= img_size
                A.PadIfNeeded(min_height=img_size, min_width=img_size,     # 補成方形
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomCrop(width=img_size, height=img_size),             # 隨機裁剪固定大小
                A.HorizontalFlip(p=0.5),                                   # 水平翻轉
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,     # 平移/縮放/旋轉
                                   rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(p=0.5),                         # 亮度對比度
                A.ColorJitter(p=0.3),                                      # 顏色抖動
                A.OneOf([
                    # 原有
                    A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.20, alpha_coef=0.04, p=0.6),
                    A.RandomRain(slant_lower=-8, slant_upper=8, drop_length=10, blur_value=3, p=0.3),
                    A.RandomSnow(brightness_coeff=1.4, snow_point_lower=0.05, snow_point_upper=0.25, p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),

                    # 模擬 Dust（顆粒 + 灰濛 + 去飽和/偏黃）
                    A.Compose([
                        A.CoarseDropout(max_holes=200, max_height=4, max_width=4, 
                                        min_holes=50, fill_value=255, p=1.0),  # 白色微粒
                        A.GaussianBlur(blur_limit=(3, 5), p=0.8),
                        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=-30, val_shift_limit=10, p=0.8),
                        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
                        A.RandomToneCurve(scale=0.1, p=0.4),
                    ], p=0.6),
                ], p=0.6),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
                A.GlassBlur(sigma=0.2, max_delta=1, iterations=1, p=0.15),

                A.MotionBlur(blur_limit=3, p=0.2),                         # 動態模糊
                A.ImageCompression(quality_lower=60, quality_upper=90, p=0.3),  # 壓縮失真
                A.Normalize(mean=(0.485, 0.456, 0.406),                    # 影像正規化（ImageNet 均值/方差）
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()                                               # 轉 tensor
            ])
        else:                                                              # 驗證/測試模式：不做隨機增強
            self.tfm = A.Compose([
                A.LongestMaxSize(max_size=img_size),                       # 最長邊 <= img_size（等比）
                A.PadIfNeeded(min_height=img_size, min_width=img_size,     # 補齊方形
                              border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=(0.485, 0.456, 0.406),                    # 同訓練正規化
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()                                               # 轉 tensor
            ])

    def __len__(self):                                                     # 回傳資料筆數
        return len(self.names)

    def __getitem__(self, i):                                              # 取得第 i 筆資料
        name = self.names[i]                                               # 檔名
        bgr = cv2.imread(str(self.img_dir/name), cv2.IMREAD_COLOR)         # 讀彩色圖（BGR）
        if bgr is None:                                                    # 防呆：檔案不存在
            raise FileNotFoundError(f"Image not found: {self.img_dir/name}")
        img = bgr[:, :, ::-1]                                              # BGR->RGB

        if self.mask_dir is not None:                                      # 若有標註
            mask = cv2.imread(str(self.mask_dir/name), cv2.IMREAD_GRAYSCALE)  # 讀灰階遮罩
            if mask is None:                                               # 防呆：遮罩不存在
                raise FileNotFoundError(f"Mask not found for {name}")
            mask = np.clip(mask, 0, 15).astype(np.uint8)                   # 保障類別落在 0..15
            out = self.tfm(image=img, mask=mask)                           # 同步增強/前處理
            return out["image"], out["mask"].long(), name                  # 回傳影像張量、遮罩張量、檔名
        else:                                                              # 無標註模式（例如測試集）
            return self.tfm(image=img)["image"], name                      # 回傳影像張量與檔名

# --------------------------
# 混淆矩陣 / mIoU
# --------------------------
def fast_hist(true, pred, num_classes):                                    # 快速建立混淆矩陣
    k = (true >= 0) & (true < num_classes)                                 # 合法像素
    return np.bincount(num_classes * true[k].astype(int) + pred[k],        # bincount 統計
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)

def compute_miou(conf_mat):                                                # 計算 mIoU 與 per-class IoU
    diag = np.diag(conf_mat).astype(np.float64)                             # TP 向量
    union = conf_mat.sum(1) + conf_mat.sum(0) - diag                        # TP + FP + FN
    iou = diag / np.maximum(union, 1e-8)                                    # IoU = TP / Union
    miou = np.nanmean(iou)                                                  # 平均 IoU
    return miou, iou

def plot_confmat(conf, save_path):                                         # 畫混淆矩陣熱圖
    plt.figure(figsize=(6,5))                                              # 圖大小
    plt.imshow(conf, interpolation='nearest')                               # 顯示矩陣
    plt.title('Confusion Matrix')                                          # 標題
    plt.xlabel('Pred')                                                     # X 軸：預測
    plt.ylabel('True')                                                     # Y 軸：真實
    plt.colorbar()                                                         # 色條
    plt.tight_layout()                                                     # 緊湊排版
    plt.savefig(save_path, dpi=150)                                        # 存檔
    plt.close()                                                            # 關閉圖避免記憶體累積

# --------------------------
# 類別權重（可緩解不平衡）
# --------------------------
def compute_class_weights(mask_dir: str, num_classes=16):                  # 計算類別權重
    hist = np.zeros(num_classes, dtype=np.int64)                            # 每類像素累加
    mask_dir = Path(mask_dir)                                              # 路徑物件
    names = [p.name for p in mask_dir.iterdir() if p.suffix.lower() == ".png"]  # 只掃 .png
    for n in names:                                                        # 逐檔計數
        m = cv2.imread(str(mask_dir/n), cv2.IMREAD_GRAYSCALE)               # 讀遮罩
        if m is None: continue                                             # 略過壞檔
        for c in range(num_classes):                                       # 逐類別數像素
            hist[c] += np.sum(m == c)
    freq = hist / max(hist.sum(), 1)                                       # 類別頻率
    inv = 1.0 / np.clip(freq, 1e-6, None)                                  # 反比（稀有類更重）
    class_weight = (inv / inv.mean()).astype(np.float32)                   # 正規化到均值=1
    return freq, class_weight                                              # 回傳頻率與權重

# --------------------------
# 驗證流程
# --------------------------
@torch.no_grad()                                                            # 驗證不需要梯度
def validate(model, loader, device, num_classes=16, return_conf=False):     # 驗證函式
    model.eval()                                                            # 切 eval 模式
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)             # 混淆矩陣
    val_loss = 0.0                                                          # 驗證損失
    ce_loss = nn.CrossEntropyLoss().to(device)                               # 使用 CE 計算 val loss

    for imgs, masks, _ in loader:                                           # 逐 batch
        imgs = imgs.to(device)                                              # 影像進裝置
        masks = masks.to(device)                                            # 標註進裝置
        logits = model(imgs)["out"]                                         # 前向（主輸出）
        val_loss += ce_loss(logits, masks).item()                           # 累加 val loss
        pred = logits.argmax(1).cpu().numpy()                               # 取類別預測（CPU Numpy）
        for t, p in zip(masks.cpu().numpy(), pred):                         # 逐張圖更新混淆矩陣
            conf += fast_hist(t.flatten(), p.flatten(), num_classes)

    miou, ious = compute_miou(conf)                                         # 計算 mIoU 與 per-class IoU
    # for cid, val in enumerate(ious):
    #     print(f"Class {cid:02d} IoU: {val:.4f}")

    acc = np.diag(conf).sum() / conf.sum().clip(min=1)                      # Overall Accuracy
    val_loss /= max(len(loader), 1)                                         # 取平均 val loss
    if return_conf:                                                         # 若需要回傳混淆矩陣
        return miou, acc, conf, ious, val_loss
    return miou, acc, conf, ious                                            # 否則只回傳指標

# --------------------------
# 主程式：訓練/驗證
# --------------------------
def main():                                                                 # 主入口
    parser = argparse.ArgumentParser()                                      # 參數解析器
    parser.add_argument("--img_dir", type=str, default="UAV_dataset/train/imgs")   # 訓練影像資料夾
    parser.add_argument("--mask_dir", type=str, default="UAV_dataset/train/masks") # 訓練遮罩資料夾
    parser.add_argument("--val_split", type=float, default=0.2)             # 驗證比例
    parser.add_argument("--img_size", type=int, default=512)                # 輸入邊長
    parser.add_argument("--batch_size", type=int, default=8)                # 批次大小
    parser.add_argument("--epochs", type=int, default=1)                   # 訓練 epoch 數
    parser.add_argument("--lr", type=float, default=3e-4)                   # 學習率
    parser.add_argument("--weight_decay", type=float, default=1e-4)         # 權重衰退
    parser.add_argument("--no_pretrain", action="store_true")               # 不用 ImageNet 預訓練
    parser.add_argument("--use_class_weight", action="store_true")          # 是否使用類別權重
    parser.add_argument("--aux_loss", action="store_true", help="開啟 DeepLabV3 輔助頭，loss += 0.4*aux")  # aux 開關
    parser.add_argument("--save_dir", type=str, default="./outputs_simple") # 輸出資料夾
    parser.add_argument("--seed", type=int, default=42)                     # 隨機種子
    parser.add_argument("--num_workers", type=int, default=4)               # DataLoader 工人數
    parser.add_argument("--cm_every", type=int, default=1, help="每幾個 epoch 存一次混淆矩陣圖")  # 混淆矩陣儲存頻率
    args = parser.parse_args()                                              # 解析參數

    set_seed(args.seed)                                                     # 固定隨機
    os.makedirs(args.save_dir, exist_ok=True)                               # 建立輸出資料夾

    if platform.system() == "Windows" and args.num_workers != 0:
        print("[info] On Windows: forcing num_workers=0 to avoid dataloader hang.")
        args.num_workers = 0
    # 類別權重（可選）
    if args.use_class_weight:                                               # 若需類別權重
        print("==> Compute class freq/weights ...")                         # 訊息
        freq, class_weight = compute_class_weights(args.mask_dir, num_classes=16)  # 統計像素比例
        print("freq:", np.round(freq, 6))                                   # 印頻率
        print("class_weight:", np.round(class_weight, 3))                   # 印權重
    else:                                                                   # 否則不使用
        class_weight = None                                                 # 設為 None

    # --------- 正確的 train/val 切分（獨立 Dataset + 同索引） ---------
    full_index_ds = SegDataset(args.img_dir, args.mask_dir, img_size=args.img_size, train=True)  # 只用來抽索引
    n_total = len(full_index_ds)                                            # 總數
    n_val = max(1, int(n_total * args.val_split))                           # 驗證數
    n_train = n_total - n_val                                               # 訓練數

    g = torch.Generator().manual_seed(args.seed)                            # 固定隨機
    perm = torch.randperm(n_total, generator=g)                             # 隨機排列索引
    val_idx = perm[:n_val]                                                  # 驗證索引
    train_idx = perm[n_val:]                                                # 訓練索引

    train_base = SegDataset(args.img_dir, args.mask_dir, img_size=args.img_size, train=True)   # 訓練 Dataset（含增強）
    val_base   = SegDataset(args.img_dir, args.mask_dir, img_size=args.img_size, train=False)  # 驗證 Dataset（不增強）

    train_set = Subset(train_base, train_idx)                               # 訓練子集
    val_set   = Subset(val_base,   val_idx)                                 # 驗證子集

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,             # 訓練 DataLoader
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,persistent_workers=False)
    val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size // 2), shuffle=False,   # 驗證 DataLoader
                            num_workers=args.num_workers, pin_memory=True,persistent_workers=False )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # 選 GPU/CPU
    print("==> Device:", device)                                            # 印裝置
    print(f"==> Batch size: {args.batch_size}, Epochs: {args.epochs}, Val split: {args.val_split} "
          f"({len(train_set)} train / {len(val_set)} val)")                 # 印 batch/epoch/切分資訊

    # 建立模型（是否載 ImageNet 預訓練、是否啟用 aux head）
    model = deeplabv3_resnet50(weights=None if args.no_pretrain else "DEFAULT",
                               aux_loss=args.aux_loss)                      # DeepLabV3 with ResNet50
    model.classifier[-1] = nn.Conv2d(256, 16, kernel_size=1)                # 主頭輸出改為 16 類
    if args.aux_loss:                                                       # 若啟用 aux head
        model.aux_classifier[-1] = nn.Conv2d(256, 16, kernel_size=1)        # 輔助頭輸出改為 16 類
    model = model.to(device)                                                # 模型搬到裝置

    # Loss（可選 class weight）
    if class_weight is not None:                                            # 帶類別權重
        weight_t = torch.tensor(class_weight, dtype=torch.float32, device=device)  # 權重張量
        main_ce = nn.CrossEntropyLoss(weight=weight_t)                      # 主頭 CE 損失
        aux_ce  = nn.CrossEntropyLoss(weight=weight_t)                      # 輔助頭 CE 損失
    else:                                                                   # 不帶權重
        main_ce = nn.CrossEntropyLoss()                                     # 主頭 CE
        aux_ce  = nn.CrossEntropyLoss()                                     # 輔助頭 CE

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)           # 餘弦退火

    scaler = torch.cuda.amp.GradScaler(enabled=True)                        # AMP 混合精度縮放器

    best_miou = -1.0                                                        # 最佳 mIoU 記錄
    best_path = Path(args.save_dir) / "best.pt"                             # 最佳模型路徑

    hist = {"epoch": [], "train_loss": [], "val_loss": [], "miou": [], "acc": [], "lr": []}  # 記錄曲線資料
    try:
        for epoch in range(1, args.epochs + 1):                                 # 訓練迴圈
            model.train()                                                       # 切訓練模式
            t0 = time.time()                                                    # 計時
            run_loss = 0.0                                                      # 累計訓練損失

            for imgs, masks, _ in train_loader:                                 # 逐 batch
                imgs = imgs.to(device); masks = masks.to(device)                # 搬到裝置

                with torch.cuda.amp.autocast(enabled=True):                     # 啟用 AMP 計算
                    out = model(imgs)                                           # 前向（可能含 aux）
                    logits = out["out"]                                         # 主頭輸出
                    loss = main_ce(logits, masks)                               # 主頭損失
                    if args.aux_loss and ("aux" in out) and (out["aux"] is not None):  # 若有 aux
                        loss = loss + 0.4 * aux_ce(out["aux"], masks)           # 加 0.4 * 輔助損失

                optimizer.zero_grad(set_to_none=True)                           # 清梯度
                scaler.scale(loss).backward()                                   # 反傳（AMP）
                scaler.step(optimizer)                                          # 參數更新（AMP）
                scaler.update()                                                 # 更新縮放因子

                run_loss += loss.item()                                         # 累加訓練損失

            miou, acc, conf, ious, val_loss = validate(model, val_loader, device, return_conf=True)  # 驗證
            if scheduler: scheduler.step()                                      # LR 調整

            lr_now = optimizer.param_groups[0]["lr"]                            # 當前學習率
            avg_train_loss = run_loss / max(len(train_loader), 1)               # 平均訓練損失

            print(f"[Epoch {epoch:03d}] "                                       # 印訓練/驗證結果
                f"train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"mIoU={miou:.4f}  Acc={acc:.4f}  "
                f"lr={lr_now:.6f}  time={(time.time()-t0):.1f}s")

            hist["epoch"].append(epoch)                                         # 紀錄 epoch
            hist["train_loss"].append(avg_train_loss)                           # 紀錄訓練損失
            hist["val_loss"].append(val_loss)                                   # 紀錄驗證損失
            hist["miou"].append(float(miou))                                    # 紀錄 mIoU
            hist["acc"].append(float(acc))                                      # 紀錄 Acc
            hist["lr"].append(float(lr_now))                                    # 紀錄學習率

            torch.save({                                                        # 存 last.pt（方便中斷續訓）
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "miou": float(miou),
                "args": vars(args)
            }, Path(args.save_dir) / "last.pt")

            if miou > best_miou:                                                # 若刷新最佳 mIoU
                best_miou = miou                                                # 更新最佳
                torch.save(model.state_dict(), best_path)                       # 存 best.pt（只權重）
                print(f"==> New best mIoU {best_miou:.4f}, saved to {best_path}")  # 提示

            if args.cm_every > 0 and (epoch % args.cm_every == 0):              # 依頻率存混淆矩陣圖
                plot_confmat(conf, Path(args.save_dir) / f"confmat_epoch{epoch:03d}.png")

            try:                                                                # 繪圖（容錯避免因字型/環境失敗中斷）
                plt.figure()                                                    # 新圖：Loss 曲線
                plt.plot(hist["epoch"], hist["train_loss"], label="train_loss") # 畫訓練 loss
                plt.plot(hist["epoch"], hist["val_loss"], label="val_loss")     # 畫驗證 loss
                plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()  # 標籤/圖例/排版
                plt.savefig(Path(args.save_dir) / "loss_curve.png", dpi=150)    # 存檔
                plt.close()                                                     # 關閉圖

                plt.figure()                                                    # 新圖：Acc / mIoU
                plt.plot(hist["epoch"], hist["acc"], label="acc")               # 畫 Acc
                plt.plot(hist["epoch"], hist["miou"], label="mIoU")             # 畫 mIoU
                plt.xlabel("epoch"); plt.ylabel("score"); plt.legend(); plt.tight_layout()  # 標籤/圖例/排版
                plt.savefig(Path(args.save_dir) / "acc_miou_curve.png", dpi=150)# 存檔
                plt.close()                                                     # 關閉圖
            except Exception as e:                                              # 捕捉繪圖錯誤
                print("[warn] plotting failed:", e)                             # 印警告

        save_dir = Path(args.save_dir)                                          # 輸出資料夾物件
    finally:
        with open(save_dir / "run_log.txt", "w", encoding="utf-8") as f:       # 寫訓練摘要
            f.write("=== Training Run Log ===\n")                               # 標頭
            f.write(time.strftime("Date: %Y-%m-%d %H:%M:%S\n"))                 # 日期時間
            f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n")  # 裝置
            f.write(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n")    # 所有參數
            f.write(f"Train/Val split: {len(train_set)} / {len(val_set)}\n")    # 切分數量
            f.write(f"Best mIoU: {best_miou:.6f}\n")                            # 最佳 mIoU

        with open(save_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:  # 寫每 epoch 指標
            writer = csv.writer(f)                                              # CSV writer
            writer.writerow(["epoch", "train_loss", "val_loss", "miou", "acc", "lr"])  # 欄名
            for i in range(len(hist["epoch"])):                                 # 逐 epoch
                writer.writerow([hist["epoch"][i], hist["train_loss"][i],       # 寫入數值
                                hist["val_loss"][i], hist["miou"][i],
                                hist["acc"][i], hist["lr"][i]])

if __name__ == "__main__":                                                  # 直接執行本檔案時
    main()                                                                  # 呼叫主程式
