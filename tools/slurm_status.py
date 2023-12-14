import os
import getpass

try:
    import tabulate
except ImportError:
    import sys
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "tabulate"])
    import tabulate

if __name__ == "__main__":
    # 1
    process = os.popen(f"squeue -o '%j %b %C %M %T %N' -u {getpass.getuser()}")
    slurm_status = process.read()
    process.close()

    process = os.popen("tmux ls")
    tmux_status = process.read()
    process.close()
    slurm_status = [line.split(" ") for line in slurm_status.split("\n")[1:-1]]
    tmux_status = [line.split(":")[0] for line in tmux_status.split("\n")[:-1]]

    for s in slurm_status:
        # gpu:x -> x
        s[1] = s[1].split(":")[-1]

    task_list = [s[0] for s in slurm_status]
    done_list = [
        # 2
        [task, "-", "-", "-", "Done", "-"]
        for task in tmux_status
        if task not in task_list
    ]

    print(
        tabulate.tabulate(
            slurm_status + done_list,
            # 3
            headers=("NAME", "GPU", "CPU", "TIME", "STATE", "NODE"),
            tablefmt="rst",
        )
    )
