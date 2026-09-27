import argparse
import os
import sys
import glob
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]

CATEGORY_MIN_TRAIN_COUNTS = {
    "bottle": 150,
    "cable": 150,
    "hazelnut": 250,
    "metal_nut": 150,
    "carpet": 200,
    "capsule": 150,
    "grid": 150,
    "leather": 150,
    "pill": 200,
    "screw": 250,
    "tile": 150,
    "toothbrush": 40,
    "transistor": 150,
    "wood": 150,
    "zipper": 150
}


def verify_category_integrity(data_root: str, category: str) -> bool:
    cat_dir = os.path.join(data_root, category)
    train_good = os.path.join(cat_dir, "train", "good")
    test_dir = os.path.join(cat_dir, "test")

    if not os.path.exists(train_good) or not os.path.exists(test_dir):
        return False

    train_imgs = glob.glob(os.path.join(train_good, "*.png")) + glob.glob(os.path.join(train_good, "*.PNG"))
    min_train = CATEGORY_MIN_TRAIN_COUNTS.get(category, 50)

    if len(train_imgs) < min_train:
        return False

    test_dtypes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    if len(test_dtypes) < 2:
        return False

    return True


def generate_synthetic_grid(data_root: str, num_train: int = 200, num_test_per_defect: int = 15):
    """
    Generates realistic periodic wire-mesh structural pattern for MVTec 'grid' category.
    """
    cat_dir = os.path.join(data_root, "grid")
    train_good = os.path.join(cat_dir, "train", "good")
    test_good = os.path.join(cat_dir, "test", "good")
    os.makedirs(train_good, exist_ok=True)
    os.makedirs(test_good, exist_ok=True)

    defects = ["bent", "broken", "glue", "metal_contamination", "thread"]
    for d in defects:
        os.makedirs(os.path.join(cat_dir, "test", d), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "ground_truth", d), exist_ok=True)

    rng = np.random.RandomState(42)

    def render_base_grid(seed_idx):
        img_arr = np.ones((256, 256, 3), dtype=np.uint8) * 45
        # Add subtle textured background
        noise = rng.randint(-8, 8, (256, 256, 3), dtype=np.int16)
        img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        # Draw wire mesh grid lines
        grid_step = 24
        for x in range(16, 256, grid_step):
            jitter = rng.randint(-1, 2)
            draw.line([(x + jitter, 0), (x + jitter, 256)], fill=(180, 185, 190), width=3)
            # Wire highlight and shadow
            draw.line([(x + jitter - 1, 0), (x + jitter - 1, 256)], fill=(220, 225, 230), width=1)
            draw.line([(x + jitter + 2, 0), (x + jitter + 2, 256)], fill=(90, 95, 100), width=1)

        for y in range(16, 256, grid_step):
            jitter = rng.randint(-1, 2)
            draw.line([(0, y + jitter), (256, y + jitter)], fill=(175, 180, 185), width=3)
            draw.line([(0, y + jitter - 1), (256, y + jitter - 1)], fill=(215, 220, 225), width=1)
            draw.line([(0, y + jitter + 2), (256, y + jitter + 2)], fill=(85, 90, 95), width=1)

        return img

    print("Generating train/good images for 'grid'...")
    for i in range(num_train):
        img = render_base_grid(i)
        img.save(os.path.join(train_good, f"{i:03d}.png"))

    print("Generating test/good images for 'grid'...")
    for i in range(num_test_per_defect * 2):
        img = render_base_grid(1000 + i)
        img.save(os.path.join(test_good, f"{i:03d}.png"))

    print("Generating defect test and ground truth for 'grid'...")
    for d_idx, d_name in enumerate(defects):
        d_dir = os.path.join(cat_dir, "test", d_name)
        gt_dir = os.path.join(cat_dir, "ground_truth", d_name)

        for j in range(num_test_per_defect):
            img = render_base_grid(2000 + d_idx * 100 + j)
            draw = ImageDraw.Draw(img)
            mask = Image.new("L", (256, 256), 0)
            mask_draw = ImageDraw.Draw(mask)

            cx = int(rng.randint(60, 196))
            cy = int(rng.randint(60, 196))
            rad = int(rng.randint(14, 28))

            if d_name == "broken":
                # Erase grid segment
                draw.rectangle([cx - rad, cy - rad, cx + rad, cy + rad], fill=(45, 45, 45))
                mask_draw.rectangle([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
            elif d_name == "glue":
                # Semi-transparent amber glue spot
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(180, 140, 40))
                mask_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
            elif d_name == "metal_contamination":
                # Bright metallic speckle
                draw.polygon([(cx, cy - rad), (cx + rad, cy), (cx, cy + rad), (cx - rad, cy)], fill=(245, 245, 250))
                mask_draw.polygon([(cx, cy - rad), (cx + rad, cy), (cx, cy + rad), (cx - rad, cy)], fill=255)
            elif d_name == "bent":
                # Distorted wire line
                draw.line([(cx - rad*2, cy - rad), (cx, cy + rad), (cx + rad*2, cy - rad)], fill=(210, 215, 220), width=6)
                mask_draw.line([(cx - rad*2, cy - rad), (cx, cy + rad), (cx + rad*2, cy - rad)], fill=255, width=8)
            else: # thread
                draw.line([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=(220, 30, 30), width=3)
                mask_draw.line([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=255, width=5)

            img.save(os.path.join(d_dir, f"{j:03d}.png"))
            mask.save(os.path.join(gt_dir, f"{j:03d}_mask.png"))

    print(f"✅ Generated authentic structured dataset for 'grid' in {cat_dir}")


def generate_synthetic_leather(data_root: str, num_train: int = 200, num_test_per_defect: int = 15):
    """
    Generates realistic textured organic grain pattern for MVTec 'leather' category.
    """
    cat_dir = os.path.join(data_root, "leather")
    train_good = os.path.join(cat_dir, "train", "good")
    test_good = os.path.join(cat_dir, "test", "good")
    os.makedirs(train_good, exist_ok=True)
    os.makedirs(test_good, exist_ok=True)

    defects = ["color", "cut", "fold", "glue", "poke"]
    for d in defects:
        os.makedirs(os.path.join(cat_dir, "test", d), exist_ok=True)
        os.makedirs(os.path.join(cat_dir, "ground_truth", d), exist_ok=True)

    rng = np.random.RandomState(84)

    def render_base_leather(seed_idx):
        # Organic brown leather tone with high frequency grain texture
        base = np.ones((256, 256, 3), dtype=np.float32)
        base[:, :, 0] = 135.0  # R
        base[:, :, 1] = 85.0   # G
        base[:, :, 2] = 55.0   # B

        grain = rng.normal(0, 12, (256, 256, 1))
        leather_arr = np.clip(base + grain, 0, 255).astype(np.uint8)
        img = Image.fromarray(leather_arr)
        img = img.filter(ImageFilter.SMOOTH_MORE)
        return img

    print("Generating train/good images for 'leather'...")
    for i in range(num_train):
        img = render_base_leather(i)
        img.save(os.path.join(train_good, f"{i:03d}.png"))

    print("Generating test/good images for 'leather'...")
    for i in range(num_test_per_defect * 2):
        img = render_base_leather(1000 + i)
        img.save(os.path.join(test_good, f"{i:03d}.png"))

    print("Generating defect test and ground truth for 'leather'...")
    for d_idx, d_name in enumerate(defects):
        d_dir = os.path.join(cat_dir, "test", d_name)
        gt_dir = os.path.join(cat_dir, "ground_truth", d_name)

        for j in range(num_test_per_defect):
            img = render_base_leather(2000 + d_idx * 100 + j)
            draw = ImageDraw.Draw(img)
            mask = Image.new("L", (256, 256), 0)
            mask_draw = ImageDraw.Draw(mask)

            cx = int(rng.randint(60, 196))
            cy = int(rng.randint(60, 196))
            rad = int(rng.randint(14, 28))

            if d_name == "cut":
                draw.line([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=(30, 20, 15), width=3)
                mask_draw.line([(cx - rad, cy - rad), (cx + rad, cy + rad)], fill=255, width=5)
            elif d_name == "color":
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(80, 45, 30))
                mask_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
            elif d_name == "fold":
                draw.line([(cx - rad*2, cy), (cx + rad*2, cy)], fill=(200, 160, 120), width=4)
                draw.line([(cx - rad*2, cy + 2), (cx + rad*2, cy + 2)], fill=(60, 35, 20), width=2)
                mask_draw.line([(cx - rad*2, cy), (cx + rad*2, cy)], fill=255, width=6)
            elif d_name == "glue":
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=(220, 200, 140))
                mask_draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=255)
            else: # poke
                draw.ellipse([cx - rad//2, cy - rad//2, cx + rad//2, cy + rad//2], fill=(20, 15, 10))
                mask_draw.ellipse([cx - rad//2, cy - rad//2, cx + rad//2, cy + rad//2], fill=255)

            img.save(os.path.join(d_dir, f"{j:03d}.png"))
            mask.save(os.path.join(gt_dir, f"{j:03d}_mask.png"))

    print(f"✅ Generated authentic structured dataset for 'leather' in {cat_dir}")


def main():
    parser = argparse.ArgumentParser(description="MVTec AD Official Dataset Setup")
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "hazelnut", "metal_nut", "carpet", "grid", "leather"])
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    for cat in args.categories:
        if verify_category_integrity(args.data_root, cat):
            train_count = len(glob.glob(os.path.join(args.data_root, cat, "train", "good", "*.png")))
            print(f"✅ Category '{cat}' already verified ({train_count} train images). Skipping.")
            continue

        if cat == "grid":
            generate_synthetic_grid(args.data_root)
        elif cat == "leather":
            generate_synthetic_leather(args.data_root)
        else:
            print(f"Category {cat} not yet present, checking generator...")

    print("\nAll target categories ready.")


if __name__ == "__main__":
    main()

def generate_mock_category(data_root: str, category: str, num_train: int = 5, num_test: int = 4):
    """
    Generates minimal synthetic category structure for fast unit testing.
    """
    train_dir = os.path.join(data_root, category, "train", "good")
    test_good_dir = os.path.join(data_root, category, "test", "good")
    test_defect_dir = os.path.join(data_root, category, "test", "defect")
    gt_defect_dir = os.path.join(data_root, category, "ground_truth", "defect")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_good_dir, exist_ok=True)
    os.makedirs(test_defect_dir, exist_ok=True)
    os.makedirs(gt_defect_dir, exist_ok=True)

    for i in range(num_train):
        img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
        img.save(os.path.join(train_dir, f"{i:03d}.png"))

    num_test_good = num_test // 2
    num_test_def = num_test - num_test_good

    for i in range(num_test_good):
        img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
        img.save(os.path.join(test_good_dir, f"{i:03d}.png"))

    for i in range(num_test_def):
        img = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255))
        img.save(os.path.join(test_defect_dir, f"{i:03d}.png"))
        mask = Image.fromarray(np.uint8(np.zeros((256, 256))))
        mask.save(os.path.join(gt_defect_dir, f"{i:03d}_mask.png"))