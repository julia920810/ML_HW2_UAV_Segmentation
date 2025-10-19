import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 路徑請改成你實際的資料夾位置
IMG_DIR = "UAV_dataset/train/imgs"
MASK_DIR = "UAV_dataset/train/masks"

# 隨機選一張影像
filenames = os.listdir(IMG_DIR)
random_file = random.choice(filenames)

# 讀取影像與對應的 mask
img_path = os.path.join(IMG_DIR, random_file)
mask_path = os.path.join(MASK_DIR, random_file)

img = cv2.imread(img_path)[:, :, ::-1]  # BGR→RGB
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 單通道灰階，值為0~15

# 建立顏色表（16類別）
# 每個類別一個固定顏色，方便可視化
colors = np.array([
    [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70],
    [102, 102, 156], [190, 153, 153], [153, 153, 153], [250, 170, 30],
    [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180],
    [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 60, 100]
], dtype=np.uint8)

# 把 mask 的每個像素轉換成 RGB 顏色
mask_color = colors[mask]
print(mask)
# 疊合顯示（調整透明度）
overlay = cv2.addWeighted(img, 0.6, mask_color, 0.4, 0)

# 顯示三張圖
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Mask (Gray)")
plt.imshow(mask, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Overlay (Image + Mask)")
plt.imshow(overlay)
plt.axis('off')

plt.suptitle(f"Sample: {random_file}", fontsize=14)
plt.tight_layout()
plt.show()
