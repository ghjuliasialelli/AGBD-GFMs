
base_text = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --error=/cluster/scratch/gsialelli/logs/pangaea-%A.txt
#SBATCH --mem-per-cpu=8G
#SBATCH --job-name=pangaea
#SBATCH --gpus=rtx_4090:1

"""

with open('../configs.txt', 'r') as f: configs = [line.strip() for line in f if line.strip()]

for config in configs :

    command = f"""torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc_per_node=1 pangaea/run.py --config-name=test ckpt_dir={config}"""

    parts = config.split('_')
    encoder = "_".join(parts[3 : -3])

    print()
    print("Encoder: ", encoder)
    print(command)
    print()

    # Write the command to a file
    with open(f"{encoder}.sh", "w") as f:
        f.write(base_text)
        f.write(command)


working_ones = ['croma_optical', 'dofa', 'gfmswin', 'prithvi', 'remoteclip', 'satlasnet_si', 'scalemae', 'ssl4eo_moco']
path_script = '/cluster/work/igp_psr/gsialelli/AGBD-GFM/pangaea/test_runs'
for encoder in working_ones:
    print(f"sbatch {path_script}/{encoder}.sh")