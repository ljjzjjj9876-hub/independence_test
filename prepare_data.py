import os
import shutil
from sklearn.model_selection import train_test_split

# ===================== 改这里 =====================
# 原始 Oxford Pet 数据集的根目录
OXFORD_ROOT = r"C:\postgraduate\python\cnn_learing\UNet\data\oxford-iiit-pet"  # 换成你本地的路径
# 你的项目 data 目录
TARGET_ROOT = "./data"
# 验证集比例
VAL_RATIO = 0.2


# =================================================

def split_and_copy(file_list, split_name):
    """把文件列表复制到目标 split 的 images/masks 里"""
    img_src_dir = os.path.join(OXFORD_ROOT, "images")
    mask_src_dir = os.path.join(OXFORD_ROOT, "annotations", "trimaps")

    img_dst_dir = os.path.join(TARGET_ROOT, split_name, "images")
    mask_dst_dir = os.path.join(TARGET_ROOT, split_name, "masks")

    os.makedirs(img_dst_dir, exist_ok=True)
    os.makedirs(mask_dst_dir, exist_ok=True)

    for name in file_list:
        # 原图是 .jpg
        src_img = os.path.join(img_src_dir, name + ".jpg")
        dst_img = os.path.join(img_dst_dir, name + ".jpg")
        shutil.copy(src_img, dst_img)

        # 标签是 .png
        src_mask = os.path.join(mask_src_dir, name + ".png")
        dst_mask = os.path.join(mask_dst_dir, name + ".png")
        shutil.copy(src_mask, dst_mask)

    print(f"{split_name} 集处理完成，共 {len(file_list)} 张")


if __name__ == "__main__":
    # 1. 读取官方 trainval 和 test 列表
    trainval_path = os.path.join(OXFORD_ROOT, "annotations", "trainval.txt")
    test_path = os.path.join(OXFORD_ROOT, "annotations", "test.txt")

    # 读取文件名（每行第一个单词是文件名）
    with open(trainval_path, "r") as f:
        trainval_names = [line.strip().split()[0] for line in f if line.strip()]
    with open(test_path, "r") as f:
        test_names = [line.strip().split()[0] for line in f if line.strip()]

    # 2. trainval 拆成 train 和 val
    train_names, val_names = train_test_split(
        trainval_names, test_size=VAL_RATIO, random_state=42
    )

    # 3. 分别复制
    split_and_copy(train_names, "train")
    split_and_copy(val_names, "val")
    split_and_copy(test_names, "test")

    print("\n全部数据整理完成！")
