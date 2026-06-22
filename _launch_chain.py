"""Launch the chain watchdog in background, then exit immediately."""
import subprocess, os

os.chdir("/root/autodl-tmp/ShaDocFormer-main")
p = subprocess.Popen(
    ["/root/miniconda3/envs/shadocformer/bin/python", "/root/autodl-tmp/ShaDocFormer-main/_chain_next.py"],
    stdout=open("/root/autodl-tmp/ShaDocFormer-main/run_chain_wrapper.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("Launched chain watchdog PID:", p.pid)
