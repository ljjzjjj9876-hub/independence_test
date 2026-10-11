import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


class SegmentationDataset(Dataset):
    def __init__(self, root_dir: str, split: str, image_size: int = 256):
        # 1. 保存传入参数
        self.split = split
        self.image_size = image_size

        # 2. 拼接图片和标签文件夹路径
        self.img_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        # 3. 获取所有图片文件名并排序，保证 image 和 mask 一一对应
        self.img_names = sorted(os.listdir(self.img_dir))

        # 4. 原图预处理流程
        self.img_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            # 从零训练可先注释掉下面这行，用预训练模型时再打开
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 5. 标签图预处理流程
        # 关键点：mask 只用最近邻插值，不做颜色归一化

        self.mask_transform = T.Compose([
            T.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST
            ),
            T.PILToTensor()
        ])

    def __len__(self):
        # 返回数据集总样本数
        return len(self.img_names)


    def __getitem__(self, idx):
        # 1. 获取当前索引对应的图片文件名
        img_name = self.img_names[idx]

        # 2. 读取原图并转 RGB 三通道
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 3. 读取对应标签图（主文件名不变，后缀换为 .png）
        mask_name = os.path.splitext(img_name)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert("L")  # 转灰度单通道

        # 4. 分别做预处理变换
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        print("img_name:", img_name)
        print("image shape:", image.hape)
        print("mask shape:", mask.shape)
        print("mask unique:", mask.unique())
        # 5. Oxford Pet 专属：三值标签转二值        # 原始值：1=前景(宠物)，2=背景，3=轮廓边缘
        # 只保留 1 作为前景，其余都算背景
        mask = (mask == 1).float()

        # 6. 返回配对好的 图片 + 标签
        return image, mask




if __name__ == "__main__":
    # ========== 测试代码：运行本文件即可验证 ==========
    dataset = SegmentationDataset(root_dir="./data", split="train", image_size=256)

    print(f"数据集总样本数：{len(dataset)}")

    img, mask = dataset[0]
    print(f"图片形状：{img.shape}")   # 预期：torch.Size([3, 256, 256])
    print(f"标签形状：{mask.shape}")  # 预期：torch.Size([1, 256, 256])
    print(f"标签唯一值：{mask.unique()}")  # 预期：只有 tensor([0., 1.])

