import subprocess, os
os.chdir("/root/autodl-tmp/ShaDocFormer-main")
p = subprocess.Popen(
    ["bash", "/root/autodl-tmp/ShaDocFormer-main/run_shadow_200ep.sh"],
    stdout=open("/root/autodl-tmp/ShaDocFormer-main/run_shadow_200ep_wrapper.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
print("Launched PID:", p.pid)
