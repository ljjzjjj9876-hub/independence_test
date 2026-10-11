from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# =========================
# 1. 数据路径
# =========================

image_dir = Path(r"C:\postgraduate\python\independence_test\data\test\images")
mask_dir = Path(r"C:\postgraduate\python\independence_test\data\test\masks")


# =========================
# 2. 随便取第一张图片
# =========================

image_paths = sorted(image_dir.glob("*.jpg"))

image_path = image_paths[0]

mask_path = mask_dir / (image_path.stem + ".png")


print("Image:", image_path.name)
print("Mask :", mask_path.name)


# =========================
# 3. 读取 image 和 mask
# =========================

image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path)


# 转 numpy，方便检查
image_np = np.array(image)
mask_np = np.array(mask)


# =========================
# 4. 检查 image
# =========================

print("\n--- Image ---")

print("shape:", image_np.shape)
print("dtype:", image_np.dtype)

print(
    "value range:",
    image_np.min(),
    "~",
    image_np.max()
)


# =========================
# 5. 检查 mask
# =========================

print("\n--- Mask ---")

print("shape:", mask_np.shape)
print("dtype:", mask_np.dtype)

print(
    "unique values:",
    np.unique(mask_np)
)


# =========================
# 6. 显示 image 和 mask
# =========================

plt.figure()

plt.imshow(image_np)
plt.title("Image")
plt.axis("off")

plt.savefig("check_image.png")
plt.close()


plt.figure()

plt.imshow(mask_np)
plt.title("Mask")
plt.axis("off")

plt.savefig("check_mask.png")
plt.close()