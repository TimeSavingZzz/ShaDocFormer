import subprocess, os
os.chdir("/root/autodl-tmp/ShaDocFormer-main")
p = subprocess.Popen(
    ["bash", "/root/autodl-tmp/ShaDocFormer-main/run_retry_queue.sh"],
    stdout=open("/root/autodl-tmp/ShaDocFormer-main/run_retry_wrapper.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("Launched retry PID:", p.pid)
