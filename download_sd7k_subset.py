"""
Download 1/4 of SD7K dataset (evenly sampled from each subfolder).
"""
import os
import openxlab
openxlab.login(ak='d7rjknrz8kkwezeneab0', sk='ajygwbpqnvpaqorvnxpqy2mzmrdwzymd29j4olne')

from openxlab.dataset.commands.utility import ContextInfoNoLogin
from openxlab.xlab.handler.user_token import trigger_update_check
from openxlab.dataset import download

trigger_update_check()
ctx = ContextInfoNoLogin()
client = ctx.get_client()
parsed_ds_name = "lkljty,ShadowDocument7K"

# Step 1: Get all files
print("Fetching file list...")
after = None
limit = 1000
all_files = []
has_more = True

while has_more:
    data_dict = client.get_api().get_dataset_files(
        dataset_name=parsed_ds_name,
        payload={},
        after=after,
        limit=limit
    )
    all_files.extend(data_dict['list'])
    has_more = data_dict.get('hasNext', False)
    if has_more:
        after = data_dict.get('after')
    print(f"  Fetched {len(all_files)} files so far...")

print(f"Total files: {len(all_files)}")

# Step 2: Group files by subfolder, EXCLUDING mask folders
from collections import defaultdict
folder_files = defaultdict(list)
for f in all_files:
    path = f['path']
    if '/mask/' in path:
        continue  # skip mask files, ShaDocFormer generates its own masks
    folder = os.path.dirname(path) or '/'
    folder_files[folder].append(path)

print("\nFolder summary (excluding mask):")
for folder, files in sorted(folder_files.items()):
    print(f"  {folder}: {len(files)} files")

# Step 3: Select 1/4 files from each folder
selected = []
for folder, files in sorted(folder_files.items()):
    files_sorted = sorted(files)
    step = 4  # take every 4th file
    subset = files_sorted[::step]
    selected.extend(subset)
    print(f"  {folder}: {len(files)} -> {len(subset)} selected")

print(f"\nTotal selected: {len(selected)} files out of {len(all_files)}")

# Step 4: Download selected files
target_base = r"F:\实验\ShaDocFormer-main\dataset\SD7K"
os.makedirs(target_base, exist_ok=True)

for i, fpath in enumerate(selected):
    print(f"[{i+1}/{len(selected)}] Downloading {fpath}...")
    try:
        download(
            dataset_repo='lkljty/ShadowDocument7K',
            source_path='/' + fpath,
            target_path=target_base
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print("\nDone!")
