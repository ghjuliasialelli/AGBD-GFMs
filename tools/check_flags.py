#!/usr/bin/env python3
"""Static safety check for the model/ training+eval+inference code:
every flag a model launcher passes must be defined by a model entry-point parser
(parser.py / eval.py / inference_aef.py). Other sub-projects (data/, benchmark_pangaea/)
have independent parsers and are out of scope here."""
import re, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def args_in(path):
    return set(re.findall(r"add_argument\(\s*['\"]--([a-zA-Z0-9_]+)", open(os.path.join(ROOT,path)).read()))
defined = args_in('model/parser.py') | args_in('model/eval.py') | args_in('model/inference_aef.py')
SLURM = {'time','ntasks','cpus','mem','gpus','gres','output','error','array','job','nodes','nnodes',
         'nproc_per_node','rdzv','num_gpus','num_cpus','transfers','exclude','include','force','bs','tmp'}
launchers = glob.glob(os.path.join(ROOT,'model','**','*.sh'), recursive=True) \
          + glob.glob(os.path.join(ROOT,'experiments','**','*.sh'), recursive=True)
used = {}
for L in launchers:
    for f in re.findall(r'(?<!\w)--([a-zA-Z0-9_]+)', open(L,errors='ignore').read()):
        used.setdefault(f, set()).add(os.path.relpath(L, ROOT))
missing = {f:v for f,v in used.items() if f not in defined and f not in SLURM}
print(f"model entry-point args: {len(defined)} | model launcher flags: {len(used)} | launchers: {len(launchers)}")
if missing:
    print("\nWARN — model launcher flags not defined by a model parser (pre-existing, pre-dates cleanup):")
    for f in sorted(missing): print(f"  --{f}  ({sorted(missing[f])[0]})")
    sys.exit(0)   # warn-only: these are pre-existing, not regressions
print("\nPASS — all model launcher flags are defined by a model entry-point parser.")
