"""
RDD Dataset Downloader for ShaDocFormer

This script downloads the RDD (Real Document Dataset) from Google Drive
and organizes it into the directory structure expected by ShaDocFormer.

Google Drive folder structure:
    RDD/
    ├── train/
    │   ├── shadow/    (→ train/input/)
    │   └── back_gt/   (→ train/target/)
    └── test/
        ├── shadow/    (→ test/input/)
        └── back_gt/   (→ test/target/)

ShaDocFormer expected structure:
    dataset/RDD/
    ├── train/
    │   ├── input/     (shadow images)
    │   └── target/    (shadow-free images)
    └── test/
        ├── input/     (shadow images)
        └── target/    (shadow-free images)

Usage:
    python download_rdd.py
"""

import os
import sys
import time
import shutil
import requests

# Google Drive folder structure
# These are the folder IDs extracted from the RDD Google Drive
FOLDERS = {
    "train_shadow": "1HjJUoA5NecK-eMPOmiylKTuDP6l2qO6u",
    "train_back_gt": "1DJxdB3wZAPJdRlQIXk5mXGt-ZLojbi3c",
    "test_shadow": "1XMftMKk_EYr_-fvYgPYvOMY72Z5qD3KA",
    "test_back_gt": "1XE1as4oPLZGx2uRAORDlJtRIqwFaCwJY",
}

# Mapping from Google Drive folder names to ShaDocFormer directories
MAPPING = {
    "train_shadow": "train/input",
    "train_back_gt": "train/target",
    "test_shadow": "test/input",
    "test_back_gt": "test/target",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "RDD")


def download_file(session, file_id, output_path, retries=3):
    """Download a single file from Google Drive with retry logic."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    for attempt in range(retries):
        try:
            response = session.get(url, stream=True, timeout=30)

            # Handle large file confirmation page
            if "confirm" in response.text and "Google Drive" in response.text:
                # Extract confirmation code
                for line in response.text.split("\n"):
                    if "confirm=" in line and "download" in line:
                        confirm_token = line.split("confirm=")[1].split("&")[0].split('"')[0]
                        url = f"https://drive.google.com/uc?export=download&confirm={confirm_token}&id={file_id}"
                        response = session.get(url, stream=True, timeout=30)
                        break

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    # Probably rate limited
                    if attempt < retries - 1:
                        wait = (attempt + 1) * 10
                        print(f"  Rate limited, waiting {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        print(f"  Failed after {retries} attempts (rate limited)")
                        return False

                with open(output_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=32768):
                        f.write(chunk)
                return True
            elif response.status_code == 429:
                wait = (attempt + 1) * 15
                print(f"  HTTP 429, waiting {wait}s...")
                time.sleep(wait)
            else:
                if attempt < retries - 1:
                    time.sleep(5)
                else:
                    print(f"  HTTP {response.status_code}")
                    return False
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(5)
            else:
                print(f"  Error: {e}")
                return False
    return False


def list_folder_files(session, folder_id):
    """List files in a Google Drive folder using the API."""
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    response = session.get(url, timeout=30)

    files = []
    # Parse file IDs from the HTML (simple regex approach)
    import re
    # Match file IDs (33-character alphanumeric strings in drive URLs)
    pattern = r'/file/d/([a-zA-Z0-9_-]{25,35})/'
    matches = re.findall(pattern, response.text)

    # Also try matching in data attributes
    pattern2 = r'data-id="([a-zA-Z0-9_-]{25,35})"'
    matches2 = re.findall(pattern2, response.text)

    all_ids = list(set(matches + matches2))
    return all_ids


def download_folder_with_gdown(folder_id, output_dir):
    """Use gdown's folder download with rate limiting."""
    import subprocess
    cmd = [
        sys.executable, "-m", "gdown",
        "--folder",
        f"https://drive.google.com/drive/folders/{folder_id}",
        "-O", output_dir,
        "--speed", "500K",  # Throttle to 500KB/s to avoid rate limits
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"gdown error: {result.stderr}")
    return result.returncode == 0


def main():
    print("=" * 60)
    print("RDD Dataset Downloader for ShaDocFormer")
    print("=" * 60)
    print()

    # Check if gdown is available
    try:
        import gdown
        print("[OK] gdown is installed")
    except ImportError:
        print("[ERROR] Please install gdown: pip install gdown")
        sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Try using gdown for each subfolder
    for src_name, folder_id in FOLDERS.items():
        dest_subdir = MAPPING[src_name]
        dest_path = os.path.join(DATASET_DIR, dest_subdir)
        os.makedirs(dest_path, exist_ok=True)

        print(f"\nDownloading {src_name} → {dest_subdir}...")
        print(f"  Folder URL: https://drive.google.com/drive/folders/{folder_id}")

        success = download_folder_with_gdown(folder_id, dest_path)

        if not success:
            print(f"  [WARNING] Some files may have failed for {src_name}")

        # Count downloaded files
        file_count = len([f for f in os.listdir(dest_path) if os.path.isfile(os.path.join(dest_path, f))])
        print(f"  Downloaded {file_count} files to {dest_subdir}")

        # Wait between folders to avoid rate limiting
        if src_name != list(FOLDERS.keys())[-1]:
            print("  Waiting 30s before next folder...")
            time.sleep(30)

    print("\n" + "=" * 60)
    print("Download complete! Verifying...")
    print("=" * 60)

    # Verify
    for subdir in ["train/input", "train/target", "test/input", "test/target"]:
        path = os.path.join(DATASET_DIR, subdir)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            print(f"  {subdir}: {count} files")
        else:
            print(f"  {subdir}: MISSING!")

    print("\nExpected: train/input=4371, train/target=4371, test/input=545, test/target=545")
    print("If numbers don't match, re-run the script or download manually from:")
    print("https://drive.google.com/drive/folders/1nS-p9qKCsFjOFyzq5q2m-Dq-PJVmLfNc")


if __name__ == "__main__":
    main()
