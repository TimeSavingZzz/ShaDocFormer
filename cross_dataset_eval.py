"""Cross-dataset generalization evaluation.
Test SD7K-trained models on RDD, and RDD-trained models on SD7K.
"""
import sys, os, json
sys.path.insert(0, '/mnt/ShaDocFormer-main')
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from train_compare_models import build_model
from data.dataset_RGB import DataReader

device = torch.device('cuda')

# Models trained on SD7K → test on RDD
SD7K_MODELS = [
    ('Restormer_SD7K',   'restormer', 'experiment_results/restormer_sd7k/restormer_best.pth'),
    ('NoSGCA_SD7K',      'shadow_guided_restormer_no_sgca', 'experiment_results/nosgca_sd7k/shadow_guided_restormer_no_sgca_best.pth'),
    ('FiLM_SD7K',        'shadow_guided_restormer_film', 'experiment_results/sgfm_sd7k/shadow_guided_restormer_film_best.pth'),
    ('Large_SD7K',       'shadow_guided_restormer_large', 'experiment_results/sglarge_sd7k/shadow_guided_restormer_large_best.pth'),
    ('Gated_SD7K',       'shadow_guided_restormer_gated', 'experiment_results/sggf_sd7k/shadow_guided_restormer_gated_best.pth'),
    ('GatedLarge_SD7K',  'shadow_guided_restormer_gated_large', 'experiment_results/sggf_large_sd7k/shadow_guided_restormer_gated_large_best.pth'),
]

# Models trained on RDD → test on SD7K (only completed ones)
RDD_MODELS = [
    ('CrossAttn_RDD',  'shadow_guided_restormer_crossattn', 'experiment_results/sgcr_rdd/shadow_guided_restormer_crossattn_best.pth'),
    ('Large_RDD',      'shadow_guided_restormer_large', 'experiment_results/sglarge_rdd/shadow_guided_restormer_large_best.pth'),
]


def evaluate(model_name, ck_path, dataset_path, split, res, model_res):
    print("\n=== %s on %s (res=%d) ===" % (model_name, dataset_path, res), flush=True)
    model, model_type = build_model(model_name, device)
    ck = torch.load(ck_path, map_location='cuda')
    model.load_state_dict(ck)
    model.eval()

    dset = DataReader(dataset_path, 'input', 'target', mode='test', ori=False,
                      img_options={'h': res, 'w': res})
    loader = DataLoader(dset, batch_size=1, shuffle=False, num_workers=0)

    psnr_list, ssim_list = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Eval ' + model_name):
            inp, gray, tar, _fname = batch
            inp, gray, tar = inp.to(device), gray.to(device), tar.to(device)

            if model_type in ('simple', 'baseline', 'textaware', 'textaware_v2'):
                out = model(inp)
            else:
                out = model(gray, inp)

            out_np = out[0].permute(1,2,0).cpu().numpy().clip(0, 1)
            tar_np = tar[0].permute(1,2,0).cpu().numpy().clip(0, 1)

            try:
                psnr_list.append(psnr(tar_np, out_np, data_range=1.0))
                ssim_list.append(ssim(tar_np, out_np, data_range=1.0, channel_axis=2,
                                      win_size=min(7, min(tar_np.shape[0], tar_np.shape[1]))))
            except Exception:
                pass

    del model, ck
    torch.cuda.empty_cache()

    avg_psnr = np.mean(psnr_list) if psnr_list else 0
    avg_ssim = np.mean(ssim_list) if ssim_list else 0
    print("  PSNR=%.2f, SSIM=%.4f (%d samples)" % (avg_psnr, avg_ssim, len(psnr_list)), flush=True)
    return {'psnr': avg_psnr, 'ssim': avg_ssim, 'n': len(psnr_list)}


print("=" * 60)
print("Cross-Dataset Generalization Evaluation")
print("=" * 60, flush=True)

results = {}

# SD7K → RDD
print("\n--- SD7K-trained models → RDD test (256x256) ---", flush=True)
for disp_name, model_name, ck in SD7K_MODELS:
    try:
        results[disp_name + '_on_RDD'] = evaluate(model_name, ck,
                                                   './dataset/RDD/test/', 'test', 256, 256)
    except Exception as e:
        print("  SKIP: %s" % str(e), flush=True)

# RDD → SD7K
print("\n--- RDD-trained models → SD7K test (320x320) ---", flush=True)
for disp_name, model_name, ck in RDD_MODELS:
    try:
        results[disp_name + '_on_SD7K'] = evaluate(model_name, ck,
                                                    './dataset/SD7K/test/', 'test', 320, 320)
    except Exception as e:
        print("  SKIP: %s" % str(e), flush=True)

print("\n\n=== Cross-Dataset Summary ===", flush=True)
print("%-24s %8s %8s" % ("Model", "PSNR", "SSIM"), flush=True)
print("-" * 42, flush=True)
for name, r in results.items():
    print("%-24s %8.2f %8.4f" % (name, r['psnr'], r['ssim']), flush=True)

with open('/mnt/ShaDocFormer-main/cross_dataset_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print("\nSaved to cross_dataset_results.json", flush=True)
print("Done.", flush=True)
