import cv2, os, numpy as np
from glob import glob
from collections import Counter
from tqdm import tqdm

MASK_DIR = "UAV_dataset/train/masks"  # 改成你的 masks 路徑
NUM_CLASSES = 16

pixel_counter = Counter()
appear_counter = [0]*NUM_CLASSES

mask_paths = sorted(glob(os.path.join(MASK_DIR, "*")))

for mp in tqdm(mask_paths, desc="分析中"):
    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    uniq, counts = np.unique(mask, return_counts=True)
    for u, c in zip(uniq, counts):
        if 0 <= u < NUM_CLASSES:
            pixel_counter[int(u)] += int(c)
            appear_counter[int(u)] += 1

total_pixels = sum(pixel_counter.values())
print("\n=== 類別統計結果 ===")
for c in range(NUM_CLASSES):
    pixel_ratio = pixel_counter[c] / total_pixels * 100 if total_pixels > 0 else 0
    print(f"class_{c:02d}: {pixel_counter[c]:,} pixels ({pixel_ratio:6.3f}%) | 出現在 {appear_counter[c]} 張圖片中")
