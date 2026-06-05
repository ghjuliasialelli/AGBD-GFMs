#!/bin/bash

# Launches all stats ablation jobs.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for script in only_2019.sh only_2020.sh only_Africa.sh only_SAm.sh only_SAs.sh \
              except_Africa.sh except_SAm.sh except_SAs.sh lite.sh ; do
    sbatch "$script_dir/$script"
done
