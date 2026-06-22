"""Wait for shadow_guided 200ep to finish, then launch the full experiment queue."""
import subprocess, os, time

os.chdir("/root/autodl-tmp/ShaDocFormer-main")

print("Waiting for shadow_guided 200ep to finish...")
while True:
    result = subprocess.run(["pgrep", "-f", "train_compare_models.*shadow_guided"], capture_output=True, text=True)
    if result.returncode != 0:
        print("shadow_guided finished!")
        break
    time.sleep(60)

print("Launching experiment queue (restormer → docdeshadower → no_sgca → concat)...")
p = subprocess.Popen(
    ["bash", "/root/autodl-tmp/ShaDocFormer-main/run_queue.sh"],
    stdout=open("/root/autodl-tmp/ShaDocFormer-main/run_queue_wrapper.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("Launched queue PID:", p.pid)
