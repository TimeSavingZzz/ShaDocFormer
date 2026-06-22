"""Batch evaluate all checkpoints for a model and output a metrics table."""
import os, sys, argparse, time, json, glob, re
import torch
import numpy as np
from train_compare_models import build_model, evaluate, DATASET_CONFIGS, build_dataloader
from models.text_aware_model import TextDetector

def eval_ckpt(checkpoint_path, model, model_type, device, val_loader, text_detector):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        ep = ckpt.get("epoch", "?")
    else:
        model.load_state_dict(ckpt, strict=False)
        ep = "?"

    metrics = evaluate(model, val_loader, text_detector, device, model_type)
    return ep, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./experiment_results_rdd/full_200ep/")
    parser.add_argument("--dataset", type=str, default="rdd")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--res", type=int, default=384)
    parser.add_argument("--epochs", type=str, default="",
                        help="Comma-separated epochs to eval, e.g. '1,10,20,200'. Default: auto-detect all")
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    variant = "v2" if args.model == "textaware" else "v1"
    model, model_type = build_model(args.model, device, model_variant=variant)
    model = model.to(device)

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    val_loader = build_dataloader(dataset_cfg, "test", False, args.batch_size, args.res)

    text_detector = TextDetector(method="opencv")  # always needed for TextPSNR eval

    # Find checkpoints
    ckpt_dir = args.checkpoint_dir
    if args.epochs:
        epochs = [int(e.strip()) for e in args.epochs.split(",")]
        files = [os.path.join(ckpt_dir, f"{args.model}_epoch{e}.pth") for e in epochs]
    else:
        pattern = os.path.join(ckpt_dir, f"{args.model}_epoch*.pth")
        files = sorted(glob.glob(pattern), key=lambda x: int(re.search(r"epoch(\d+)", x).group(1)))

    print(f"Evaluating {len(files)} checkpoints for {args.model} ({model_type})")
    results = {}
    for fpath in files:
        ep = int(re.search(r"epoch(\d+)", fpath).group(1))
        if ep > 200:
            continue
        print(f"  E{ep}...", end=" ", flush=True)
        ep_out, metrics = eval_ckpt(fpath, model, model_type, device, val_loader, text_detector)
        results[str(ep)] = {
            "psnr": round(float(metrics["psnr"]), 2),
            "ssim": round(float(metrics["ssim"]), 4),
            "rmse": round(float(metrics["rmse"]), 2),
            "text_psnr": round(float(metrics["text_psnr"]), 2),
        }
        print(f"PSNR={results[str(ep)]['psnr']:.2f} SSIM={results[str(ep)]['ssim']:.4f} TextPSNR={results[str(ep)]['text_psnr']:.2f}")

    out_path = args.output_json or os.path.join(ckpt_dir, f"eval_{args.model}_all.json")
    out = {
        "model": args.model,
        "model_type": model_type,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Best PSNR: {max(r['psnr'] for r in results.values()):.2f} at E{max(results, key=lambda k: results[k]['psnr'])}")

if __name__ == "__main__":
    main()
