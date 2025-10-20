# infer_seg16.py
# ------------------------------------------------------------
# 使用訓練出的 best.pt 進行推論，輸出：
# 1) 灰階 PNG（像素值 0..15）
# 2) (可選) RLE CSV（每類一欄；若該類別不存在填 "none"）
# ------------------------------------------------------------

import os
from pathlib import Path
import argparse
import csv
import numpy as np
import cv2

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

import albumentations as A
from albumentations.pytorch import ToTensorV2

# RLE 編碼（For Kaggle 格式）
def rle_encode(bin_mask: np.ndarray) -> str:
    pixels = bin_mask.flatten(order='F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[:-1:2]
    return ' '.join(map(str, runs))

def build_infer_transform(img_size):
    # 等比縮放 + Pad 到 img_size；推論 Normalize
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size,
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std =(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

@torch.no_grad()
def infer_one(model, img_rgb, device, tfm, orig_h, orig_w):
    # 前處理
    out = tfm(image=img_rgb)
    x = out["image"].unsqueeze(0).to(device)

    # 推論
    logits = model(x)["out"]
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    # 去掉 Pad（先反推實際貼上位置）
    # 因為我們只 Pad 到正方形，縮放比例是相同的；直接把 pred Resize 回原圖大小即可。
    pred_full = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return pred_full

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="UAV_dataset/test",
                        help="測試影像資料夾（無標註）")
    parser.add_argument("--weights", type=str, default="./outputs_simple/best.pt",
                        help="訓練存下的 best.pt")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--save_pred_png", action="store_true",
                        help="若開啟則輸出灰階 PNG")
    parser.add_argument("--pred_dir", type=str, default="./outputs_simple/preds")
    parser.add_argument("--gen_rle_csv", action="store_true",
                        help="若開啟則輸出 RLE CSV")
    parser.add_argument("--rle_csv", type=str, default="./outputs_simple/sample_submission.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==> Device:", device)

    # 建立模型（輸出 16 類）
    model = deeplabv3_resnet50(weights=None)
    model.classifier[-1] = nn.Conv2d(256, 16, kernel_size=1)
    # 載入權重
    state = torch.load(args.weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        # 可能是直接存 model.state_dict()
        try:
            model.load_state_dict(state)
        except:
            # 也可能是 last.pt 格式
            model.load_state_dict(state.get("model", state))
    else:
        model.load_state_dict(state)
    model = model.to(device).eval()

    tfm = build_infer_transform(args.img_size)

    # 列出測試影像
    test_dir = Path(args.test_dir)
    names = sorted([p.name for p in test_dir.iterdir()
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])

    # 輸出資料夾
    pred_dir = Path(args.pred_dir); 
    if args.save_pred_png: pred_dir.mkdir(parents=True, exist_ok=True)

    # 若要 RLE
    if args.gen_rle_csv:
        csv_fp = open(args.rle_csv, "w", newline="")
        writer = csv.writer(csv_fp)
        header = ["img"] + [f"class_{i}" for i in range(16)]
        writer.writerow(header)

    for i, name in enumerate(names, 1):
        # 讀 RGB
        img = cv2.imread(str(test_dir/name), cv2.IMREAD_COLOR)[:, :, ::-1]
        h, w = img.shape[:2]

        # 推論
        pred = infer_one(model, img, device, tfm, h, w)

        # PNG
        if args.save_pred_png:
            cv2.imwrite(str(pred_dir/(Path(name).stem + ".png")), pred)

        # RLE
        if args.gen_rle_csv:
            row = [name]
            for cid in range(16):
                binmask = (pred == cid).astype(np.uint8)
                row.append("none" if binmask.sum() == 0 else rle_encode(binmask))
            writer.writerow(row)

        if i % 20 == 0:
            print(f"Infer {i}/{len(names)} ...")

    if args.gen_rle_csv:
        csv_fp.close()
        print(f"==> Kaggle CSV saved: {args.rle_csv}")

    print("==> Inference done.")

if __name__ == "__main__":
    main()
