"""Pre-compute MSER text attention maps for all training images using CPU multiprocessing."""
import os, sys, argparse
import cv2
import numpy as np
import torch
from multiprocessing import Pool
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def compute_attention(img_path):
    try:
        mser = cv2.MSER_create()
        mser.setDelta(5)
        mser.setMinArea(60)
        mser.setMaxArea(14400)
        mser.setMaxVariation(0.25)
        mser.setMinDiversity(0.2)
        mser.setMaxEvolution(200)
        mser.setMinMargin(0.003)
        mser.setEdgeBlurSize(5)
        img = cv2.imread(img_path)
        if img is None:
            return os.path.basename(img_path), None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        regions, _ = mser.detectRegions(gray)
        mask = np.zeros(gray.shape, dtype=np.float32)
        if regions is not None:
            for region in regions:
                if len(region) > 0:
                    hull = cv2.convexHull(region.reshape(-1, 1, 2))
                    cv2.fillConvexPoly(mask, hull, 10.0)
        mask = np.clip(mask / 10.0, 0.0, 1.0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return os.path.basename(img_path), torch.from_numpy(mask).float()
    except Exception:
        return os.path.basename(img_path), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--target_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for subdir, name in [(args.input_dir, 'input'), (args.target_dir, 'target')]:
        files = sorted([f for f in os.listdir(subdir) if is_image_file(f)])
        paths = [os.path.join(subdir, f) for f in files]
        out_subdir = os.path.join(args.output_dir, name)
        os.makedirs(out_subdir, exist_ok=True)

        print(f"Processing {len(paths)} images from {subdir} with {args.num_workers} workers...")

        with Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap(compute_attention, paths), total=len(paths), desc=name))

        saved = 0
        for fname, attn in results:
            if attn is not None:
                out_path = os.path.join(out_subdir, fname.rsplit('.', 1)[0] + '.pt')
                torch.save(attn, out_path)
                saved += 1
        print(f"  -> Saved {saved} attention maps to {out_subdir}")

    print("Done!")


if __name__ == '__main__':
    main()
