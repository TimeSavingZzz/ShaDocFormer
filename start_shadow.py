#!/usr/bin/env python3
"""Kill zombie processes and start shadow_guided training."""
import os, subprocess, time

os.chdir("/root/autodl-tmp/ShaDocFormer-main")

# Kill zombies
os.system("pkill -9 -f 'bash -c cd.*train_compare.*v1' 2>/dev/null")
os.system("pkill -9 -f queue_shadow_guided 2>/dev/null")
time.sleep(2)

# Launch
with open("run_shadow_guided_wrapper.log", "w") as f:
    subprocess.Popen(
        ["bash", "run_shadow_guided.sh"],
        stdout=f, stderr=subprocess.STDOUT,
        start_new_session=True,
    )

time.sleep(10)

# Verify
result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True)
print("GPU:", result.stdout.strip())

# Check log
try:
    with open("experiment_results_rdd/full_200ep/train_shadow_guided_50ep.log") as log:
        lines = log.readlines()
        for l in lines[:15]:
            print(l.rstrip())
except FileNotFoundError:
    print("Log not ready yet")

# Check running
result = subprocess.run(["pgrep", "-f", "train_compare_models.*shadow_guided"], capture_output=True)
if result.returncode == 0:
    print("OK - shadow_guided is running (PIDs: {})".format(result.stdout.decode().strip().replace("\n", " ")))
else:
    print("WARNING: shadow_guided may not have started")
