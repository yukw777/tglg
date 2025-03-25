from pathlib import Path

from jsonargparse import auto_cli
from lightning_sdk import Job, Machine, Studio


def main(
    studio_user: str,
    studio_name: str,
    teamspace: str,
    frame_token_interval_thresholds: list[float],
    root_results_dir: Path,
    run_name_prefix: str,
    cwd: str = ".",
    run_inference_args: dict[str, str] | None = None,
    dry_run: bool = False,
    machine: str = "L40S_X_4",
    interruptible: bool = True,
) -> None:
    torchrun_map = {
        Machine.L40S: "--nnodes=1 --nproc_per_node=1",
        Machine.L40S_X_4: "--nnodes=1 --nproc_per_node=4",
    }
    selected_machine = getattr(Machine, machine)
    assert selected_machine in torchrun_map, f"{machine} is not supported."

    if not dry_run:
        studio = Studio(user=studio_user, name=studio_name, teamspace=teamspace)
        studio.start()

    script_cmd = f"cd {cwd} && torchrun {torchrun_map[selected_machine]} scripts/run_inference.py \\"
    if run_inference_args is None:
        run_inference_args = {}
    for i, threshold in enumerate(frame_token_interval_thresholds):
        run_name = f"{run_name_prefix}+threshold={threshold}"
        run_inference_args["model.frame_token_interval_threshold"] = str(threshold)
        run_inference_args["results_dir"] = str(root_results_dir / run_name)
        run_inference_args["wandb_run_name"] = run_name
        script_args = " \\\n".join(f"--{k} {v}" for k, v in run_inference_args.items())
        cmd = "\n".join([script_cmd, script_args])
        print(f"Run {i}:")
        print(cmd)
        print()
        if not dry_run:
            Job.run(
                command=cmd,
                name=run_name,
                machine=selected_machine,
                studio=studio,
                interruptible=interruptible,
            )


if __name__ == "__main__":
    auto_cli(main, as_positional=False)
