import subprocess

from jsonargparse import auto_cli


def main(
    job_name: str,
    account: str,
    partition: str,
    time: str,
    num_gpus: int,
    mem_per_gpu: str,
    wandb_project: str,
    num_dataloader_workers: int,
    email: str | None = None,
    hf_home: str | None = None,
    dry_run: bool = False,
    train_args: dict[str, str] | None = None,
) -> None:
    if email is None:
        email = ""
    else:
        email = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=BEGIN,END"
    if hf_home is None:
        hf_home = ""
    else:
        hf_home = f"export HF_HOME={hf_home}"

    multi_gpu = f"""RDZV_ID=$RANDOM
MASTER_NODE=$(srun --nodes=1 --ntasks=1 hostname)
srun --cpus-per-task {num_dataloader_workers} uv run torchrun \\
--nnodes={num_gpus} \\
--nproc_per_node=1 \\
--rdzv-id=$RDZV_ID \\
--rdzv-backend=c10d \\
--rdzv-endpoint=$MASTER_NODE \\
scripts/train_videollm_online_ego4d_goalstep.py \\"""
    single_gpu = "uv run python scripts/train_videollm_online_ego4d_goalstep.py \\"

    if train_args is None:
        train_args = {}
    args = " \\\n".join(f"--{k} {v}" for k, v in train_args.items())
    script = rf"""#!/bin/bash

#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --job-name=train_videollm_online_ego4d_goalstep:{job_name}
{email}
#SBATCH --account={account}
#SBATCH --ntasks={num_gpus}
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task={num_dataloader_workers}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --output=%x-%j.log

module load cuda gcc
# https://github.com/dmlc/decord/issues/156
export DECORD_EOF_RETRY_MAX=20480
{hf_home}
export WANDB_NAME={job_name}
export WANDB_PROJECT={wandb_project}
{single_gpu if num_gpus < 2 else multi_gpu}
--num_dataloader_workers {num_dataloader_workers} \
{args}
"""
    print(script)
    if not dry_run:
        subprocess.run(["sbatch"], input=script, text=True)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
