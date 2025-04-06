import subprocess

from jsonargparse import auto_cli


def main(
    job_name: str,
    account: str,
    partition: str,
    time: str,
    total_gpus: int,
    gpus_per_node: int,
    mem_per_gpu: str,
    wandb_project: str,
    num_dataloader_workers: int,
    email: str | None = None,
    hf_home: str | None = None,
    dry_run: bool = False,
    run_inference_args: dict[str, str] | None = None,
) -> None:
    # Validate that total_gpus is evenly divisible by gpus_per_node.
    if total_gpus % gpus_per_node != 0:
        raise ValueError(
            f"Total GPUs ({total_gpus}) must be evenly divisible by GPUs per node ({gpus_per_node})."
        )
    # Compute the number of nodes required.
    nodes = total_gpus // gpus_per_node

    if email is None:
        email = ""
    else:
        email = f"#SBATCH --mail-user={email}\n#SBATCH --mail-type=BEGIN,END"
    if hf_home is None:
        hf_home = ""
    else:
        hf_home = f"export HF_HOME={hf_home}"

    multi_gpu = f"""RDZV_ID=$RANDOM
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=$((1024 + RANDOM % 64510))
MP_MANAGER_PORT=$((MASTER_PORT + 1))
srun uv run torchrun \\
--nnodes={nodes} \\
--nproc_per_node={gpus_per_node} \\
--rdzv-id=$RDZV_ID \\
--rdzv-backend=c10d \\
--rdzv-endpoint=$MASTER_NODE:$MASTER_PORT \\
scripts/run_inference.py \\"""
    single_gpu = "uv run python scripts/run_inference.py \\"

    if run_inference_args is None:
        run_inference_args = {}
    run_inference_args["num_dataloader_workers"] = str(num_dataloader_workers)
    run_inference_args["wandb_project"] = wandb_project
    run_inference_args["wandb_run_name"] = job_name
    run_inference_args["mp_manager_ip_addr"] = "$MASTER_NODE"
    run_inference_args["mp_manager_port"] = "$MP_MANAGER_PORT"
    args = " \\\n".join(f"--{k} {v}" for k, v in run_inference_args.items())
    script = rf"""#!/bin/bash

#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --job-name=run-inference:{job_name}
{email}
#SBATCH --account={account}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task={num_dataloader_workers * gpus_per_node}
#SBATCH --mem-per-gpu={mem_per_gpu}
#SBATCH --output=%x-%j.log

module load cuda gcc
# https://github.com/dmlc/decord/issues/156
export DECORD_EOF_RETRY_MAX=20480
{hf_home}
{single_gpu if total_gpus < 2 else multi_gpu}
{args}
"""
    print(script)
    if not dry_run:
        subprocess.run(["sbatch"], input=script, text=True)


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
