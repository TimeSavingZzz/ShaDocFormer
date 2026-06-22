#!/usr/bin/env python3
"""Auto-monitor v2 training. Detects and handles failures automatically."""
import os, sys, time, subprocess, glob

WORKDIR = "/root/autodl-tmp/ShaDocFormer-main"
LOG_FILE = os.path.join(WORKDIR, "train_v2_run3.log")
OUTPUT_DIR = os.path.join(WORKDIR, "experiment_results_rdd/full_200ep")
MONITOR_LOG = os.path.join(WORKDIR, "monitor_v2.log")

CMD = (
    f"cd {WORKDIR} && "
    f"nohup /root/miniconda3/envs/shadocformer/bin/python -u train_compare_models.py "
    f"--model textaware --model_variant v2 --epochs 200 --batch_size 4 "
    f"--dataset rdd --output ./experiment_results_rdd/full_200ep/ "
    f"> train_v2_run3.log 2>&1 &"
)


def log(msg):
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(MONITOR_LOG, 'a') as f:
        f.write(line + "\n")


def get_training_pid():
    try:
        out = subprocess.check_output(
            "ps aux | grep 'train_compare_models.py.*textaware' | grep -v grep | grep -v monitor",
            shell=True, timeout=10
        ).decode().strip()
        for line in out.split("\n"):
            if "python" in line and "train_compare" in line:
                return int(line.split()[1])
    except:
        pass
    return None


def get_last_epoch():
    """Parse log for latest epoch eval."""
    try:
        with open(LOG_FILE, 'r') as f:
            content = f.read()
        lines = content.split("\n")
        last_epoch = 0
        last_textpsnr = None
        for line in lines:
            if line.strip().startswith("[") and "PSNR=" in line and "TextPSNR=" in line:
                parts = line.strip().split()
                ep_str = parts[0].split("/")[0].replace("[", "")
                try:
                    last_epoch = int(ep_str)
                except:
                    pass
                for p in parts:
                    if p.startswith("TextPSNR="):
                        try:
                            last_textpsnr = float(p.split("=")[1])
                        except:
                            pass
        return last_epoch, last_textpsnr
    except:
        return 0, None


def get_loss_issues():
    """Check last 500 chars for NaN."""
    try:
        with open(LOG_FILE, 'r') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 2000))
            tail = f.read()
        if "NaN" in tail or "nan" in tail:
            return True
    except:
        pass
    return False


def backup_and_clean(run_name="auto_backup"):
    """Move existing checkpoints to backup dir."""
    bkdir = os.path.join(OUTPUT_DIR, run_name)
    os.makedirs(bkdir, exist_ok=True)
    for pattern in ["textaware_epoch*.pth", "textaware_best.pth"]:
        for f in glob.glob(os.path.join(OUTPUT_DIR, pattern)):
            os.rename(f, os.path.join(bkdir, os.path.basename(f)))


def restart_training():
    """Kill old and start fresh."""
    pid = get_training_pid()
    if pid:
        try:
            os.kill(pid, 9)
        except:
            pass
        time.sleep(3)
    subprocess.Popen(CMD, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(5)
    new_pid = get_training_pid()
    return new_pid


def check_gpu_oom():
    """Check if GPU OOM occurred."""
    try:
        with open(LOG_FILE, 'r') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 3000))
            tail = f.read()
        if "CUDA out of memory" in tail or "OOM" in tail:
            return True
    except:
        pass
    return False


def main():
    log("=== V2 Monitor started ===")
    consecutive_fails = 0
    last_textpsnr_zero = 0  # epoch when first zero detected
    restart_count = 0
    max_restarts = 3

    while True:
        try:
            pid = get_training_pid()
            epoch, textpsnr = get_last_epoch()
            has_nan = get_loss_issues()
            is_oom = check_gpu_oom()

            status_parts = [f"E{epoch}"]
            if textpsnr is not None:
                status_parts.append(f"TextPSNR={textpsnr:.2f}")

            if pid:
                status_parts.append(f"PID={pid}")
            else:
                status_parts.append("MISSING")
                consecutive_fails += 1
            if has_nan:
                status_parts.append("NaN!")
            if is_oom:
                status_parts.append("OOM!")

            log(" | ".join(status_parts))

            should_restart = False
            reason = ""

            # Condition 1: Process dead
            if not pid:
                should_restart = True
                reason = "process dead"

            # Condition 2: NaN loss
            elif has_nan:
                should_restart = True
                reason = "NaN loss"

            # Condition 3: TextPSNR=0 after E5 (detector collapsed)
            elif epoch >= 5 and textpsnr is not None and textpsnr < 0.01:
                if last_textpsnr_zero == 0:
                    last_textpsnr_zero = epoch
                elif epoch - last_textpsnr_zero >= 3:
                    should_restart = True
                    reason = f"TextPSNR=0 persisted for {epoch - last_textpsnr_zero} epochs"
            else:
                last_textpsnr_zero = 0

            # Condition 4: OOM
            if is_oom and not should_restart:
                should_restart = True
                reason = "OOM detected, retry with lower bs"

            if should_restart and restart_count < max_restarts:
                log(f"ACTION: Restarting ({reason}). Count={restart_count+1}/{max_restarts}")
                backup_and_clean(f"crash_backup_{restart_count + 1}")
                new_pid = restart_training()
                log(f"Restarted with PID={new_pid}")
                restart_count += 1
                consecutive_fails = 0
                last_textpsnr_zero = 0
            elif should_restart:
                log(f"FATAL: Max restarts ({max_restarts}) reached. Giving up.")

        except Exception as e:
            log(f"Monitor error: {e}")

        time.sleep(300)  # Check every 5 minutes


if __name__ == "__main__":
    main()
