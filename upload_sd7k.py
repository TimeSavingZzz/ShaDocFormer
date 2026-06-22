"""Upload SD7K dataset to server via SFTP."""
import paramiko, os, sys, pathlib

host = 'connect.bjb1.seetacloud.com'
port = 14497
user = 'root'
pw = 'cC9nNoS+7QqM'

local_base = str(pathlib.Path(r'C:\迅雷下载\实验\ShaDocFormer-main\dataset\SD7K\lkljty___ShadowDocument7K'))
remote_base = '/root/autodl-tmp/ShaDocFormer-main/dataset/SD7K/lkljty___ShadowDocument7K'

t = paramiko.Transport((host, port))
t.connect(username=user, password=pw)
sftp = paramiko.SFTPClient.from_transport(t)

# Create remote dirs
for sub in ['train/input', 'train/target', 'test/input', 'test/target']:
    remote_path = remote_base + '/' + sub
    try:
        sftp.mkdir(remote_path)
    except:
        pass

# Count total
total = 0
for root, dirs, files in os.walk(local_base):
    total += len(files)
print(f'Uploading {total} files...')

uploaded = 0
for root, dirs, files in os.walk(local_base):
    rel = os.path.relpath(root, local_base).replace('\\', '/')
    remote_dir = remote_base
    if rel != '.':
        remote_dir = remote_base + '/' + rel
    for f in files:
        local_path = os.path.join(root, f)
        remote_path = remote_dir + '/' + f
        try:
            sftp.put(local_path, remote_path)
            uploaded += 1
            if uploaded % 300 == 0:
                print(f'{uploaded}/{total} ({uploaded*100/total:.1f}%)')
        except Exception as e:
            print(f'Error {f}: {e}')

sftp.close()
t.close()
print(f'Done: {uploaded}/{total} files uploaded')
