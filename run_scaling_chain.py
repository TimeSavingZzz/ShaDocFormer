"""Chain-launch remaining scaling curve jobs sequentially on GPU 1."""
import subprocess, os, sys
from datetime import datetime

JOBS = [
    ('shadow_guided_restormer_film', 'sd7k', 'film', 60),
    ('shadow_guided_restormer_gated', 'sd7k', 'gated', 60),
    ('shadow_guided_restormer_no_sgca', 'sd7k', 'concat', 100),
    ('shadow_guided_restormer_film', 'sd7k', 'film', 100),
    ('shadow_guided_restormer_gated', 'sd7k', 'gated', 100),
    ('shadow_guided_restormer_no_sgca', 'sd7k', 'concat', 150),
    ('shadow_guided_restormer_film', 'sd7k', 'film', 150),
    ('shadow_guided_restormer_gated', 'sd7k', 'gated', 150),
]

BASE_DIR = '/mnt/ShaDocFormer-main'
PYTHON = '/opt/miniconda3/envs/shadocformer/bin/python3'

for model, dataset, tag, size in JOBS:
    cmd = [
        PYTHON, '-u', 'train_compare_models.py',
        '--model', model, '--dataset', dataset,
        '--epochs', '100', '--lr', '2e-4',
        '--batch_size', '1', '--res', '320',
        '--train_dir', f'dataset/SD7K_subsets/sd7k_{size}/train/',
        '--val_dir', f'dataset/SD7K_subsets/sd7k_{size}/test/',
        '--output', f'experiment_results/scaling_{tag}_{size}',
    ]
    log_path = f'/mnt/scaling_{tag}_{size}.log'
    now = datetime.now().strftime('%H:%M')
    print(f'[{now}] Launching {tag} @ {size} pairs...')
    sys.stdout.flush()
    with open(log_path, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=BASE_DIR,
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': '1'})
    now = datetime.now().strftime('%H:%M')
    print(f'[{now}] Done: {tag} @ {size}')
    sys.stdout.flush()

print('All scaling curve jobs complete!')
