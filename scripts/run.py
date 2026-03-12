import datetime as dt
from os import mkdir, system
from subprocess import PIPE, Popen
from time import sleep

HOSTS_CNT = 17

V100S_32G = [1, 2, 3, 4, 5, 6] + [11, 12]
V100_32G = [7, 8]
V100_16G = [9, 10]
A100_40G = [13]
L20_48G = [14, 15, 16, 17, 18]

ALL = list(range(1, HOSTS_CNT + 1))
SMALL = V100_16G
MEDIUM = V100S_32G + V100_32G
LARGE = A100_40G + L20_48G

ORDER = "r15s"
INCLUDE = SMALL
EXCLUDE = None
INTERVAL = 15

GPU_MODE = "exclusive_process"  # shared, exclusive_process


def is_job_pending():
    res = Popen("bjobs -p", shell=True, stdout=PIPE, stderr=PIPE).communicate()[1]
    return b"No pending job found" not in res


def get_hosts():
    in_hosts = set(INCLUDE) if INCLUDE else range(1, HOSTS_CNT + 1)
    ex_hosts = set(EXCLUDE) if EXCLUDE else set()

    in_hosts = set(in_hosts) - ex_hosts
    ex_hosts = set(range(1, HOSTS_CNT + 1)) - in_hosts

    if len(in_hosts) > len(ex_hosts):
        return False, ex_hosts
    else:
        return True, in_hosts


exp = {
    "5": [
        "--subset_n 5",
        "--model mage_vit_base_patch16",
        "--data_path ~/data/tiny-imagenet-200",
        "--epochs 10000",
        "--batch_size 1",
        "--smoothing 0.0",
        "--no_share_embedding",
        "--disable_aug",
        "--dropout 0.0",
        "--mask_ratio_mu 0.5",
        "--mask_ratio_std 0.5",
        "--mask_ratio_min 0.0",
        "--mask_ratio_max 1.0",
    ],
    "smooth0.1": [
        "--subset_n 1",
        "--model mage_vit_base_patch16",
        "--data_path ~/data/tiny-imagenet-200",
        "--epochs 10000",
        "--batch_size 1",
        "--smoothing 0.1",
        "--no_share_embedding",
        "--disable_aug",
        "--dropout 0.0",
        "--mask_ratio_mu 0.5",
        "--mask_ratio_std 0.5",
        "--mask_ratio_min 0.0",
        "--mask_ratio_max 1.0",
    ],
    "share_embed": [
        "--subset_n 1",
        "--model mage_vit_base_patch16",
        "--data_path ~/data/tiny-imagenet-200",
        "--epochs 10000",
        "--batch_size 1",
        "--smoothing 0.0",
        "--disable_aug",
        "--dropout 0.0",
        "--mask_ratio_mu 0.5",
        "--mask_ratio_std 0.5",
        "--mask_ratio_min 0.0",
        "--mask_ratio_max 1.0",
    ],
    "smooth0.1_share_embed": [
        "--subset_n 1",
        "--model mage_vit_base_patch16",
        "--data_path ~/data/tiny-imagenet-200",
        "--epochs 10000",
        "--batch_size 1",
        "--smoothing 0.1",
        "--disable_aug",
        "--dropout 0.0",
        "--mask_ratio_mu 0.5",
        "--mask_ratio_std 0.5",
        "--mask_ratio_min 0.0",
        "--mask_ratio_max 1.0",
    ],
}

for i, (name, exp_args) in enumerate(exp.items(), start=1):
    print(f"Processing {i}/{len(exp)}\n")

    wait_cnt = 0
    while is_job_pending():
        wait_cnt += 1
        print(f"There are pending jobs, waiting... ({wait_cnt * INTERVAL}s elapsed)\n")
        sleep(INTERVAL)

    log_name = dt.datetime.now().strftime("%y%m%d-%H%M%S") + f"_{name}"
    mkdir(f"output_dir/train/{log_name}")
    exp_args.append(f"--output_dir output_dir/train/{log_name}")
    cmd = f"python3 main_pretrain.py {' '.join(exp_args)}"

    use_in_hosts, hosts = get_hosts()
    select_list = [f"hname{'==' if use_in_hosts else '!='}gpu{gid:02}" for gid in hosts]
    bsub_args = [
        f"-gpu 'num=1:mode={GPU_MODE}'",
        f"-R 'order[{ORDER}]'",
        f"-J {name}",
        f"-oo output_dir/train/{log_name}/{name}.out",
        f"-eo output_dir/train/{log_name}/{name}.err",
    ]
    if select_list:
        bsub_args.append(
            f"-R 'select[{(' || ' if use_in_hosts else ' && ').join(select_list)}]'"
        )

    full_cmd = f"bsub {' '.join(bsub_args)} {cmd}"
    print(f"{full_cmd}\n")
    system(full_cmd)
    print()

    if i == len(exp):
        break
    sleep(INTERVAL)

    system("bjobs | sort -k1,1n")
    print()
