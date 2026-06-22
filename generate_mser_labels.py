"""Generate MSER pseudo-labels for DTRM supervision. Run on server."""
import os, sys, cv2, torch
import numpy as np
from PIL import Image
from tqdm import tqdm

def generate_one(img_path, mser):
    img = np.array(Image.open(img_path).convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros(gray.shape, dtype=np.float32)
    if regions is not None:
        for region in regions:
            if len(region) > 0:
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                cv2.fillConvexPoly(mask, hull, 10.0)
    mask = np.clip(mask / 10.0, 0.0, 1.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 5))
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    return torch.from_numpy(mask).float().unsqueeze(0)

def main():
    dataset_root = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/ShaDocFormer-main/dataset/RDD'
    tar_dir = os.path.join(dataset_root, 'train', 'gt')
    out_dir = os.path.join(dataset_root, 'attention', 'target')
    os.makedirs(out_dir, exist_ok=True)

    img_files = sorted([f for f in os.listdir(tar_dir) if f.lower().endswith(('jpg','jpeg','png'))])
    mser = cv2.MSER_create()

    for fname in tqdm(img_files, desc="MSER labels"):
        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.pt')
        if os.path.exists(out_path):
            continue
        mask = generate_one(os.path.join(tar_dir, fname), mser)
        torch.save(mask, out_path)
    print(f"Done. Generated labels in {out_dir}")

if __name__ == '__main__':
    main()
