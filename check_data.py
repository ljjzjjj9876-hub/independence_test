from pathlib import Path

image_dir = Path(r"C:\postgraduate\python\independence_test\data\test\images")
mask_dir = Path(r"C:\postgraduate\python\independence_test\data\test\masks")

images = list(image_dir.glob("*.jpg"))
masks = list(mask_dir.glob("*.png"))

print("Images:", len(images))
print("Masks:", len(masks))

images = sorted(image_dir.glob("*.jpg"))

for image_path in images[:10]:

    mask_path = mask_dir / (
        image_path.stem + ".png"
    )

    print(
        image_path.name,
        "->",
        mask_path.name,
        mask_path.exists()
    )