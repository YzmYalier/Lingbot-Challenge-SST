#!/bin/bash
# Re-derive the longest-first task order in start_robotwin_infer_and_eval.sh.
#
# Slots claim tasks from the head of task_list_all, so putting the slowest tasks
# first (the LPT rule) keeps the tail short. The baked-in order was measured once;
# re-run this if per-task cost shifts a lot (new model or hardware)
# and paste the output over task_list_all.
#
# Usage: bash rank_tasks_by_duration.sh <run_dir> [<run_dir> ...]
#
# Each run_dir needs a stats.txt (written at the end of a completed run). Pass
# several to average out one run's noise. Tasks are ranked by MEAN RANK, not mean
# seconds: total time differs between task_configs (demo_clean is much faster than
# demo_randomized) while the relative order is stable, so averaging raw seconds
# would let the slowest config dominate.
set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <run_dir> [<run_dir> ...]" >&2
    exit 2
fi

for d in "$@"; do
    if [ ! -f "${d}/stats.txt" ]; then
        echo "Error: ${d}/stats.txt not found (run incomplete?)" >&2
        exit 2
    fi
done

python3 - "$@" <<'PY'
import re, sys

runs = sys.argv[1:]
per_run = []
for d in runs:
    dur = {}
    for line in open(f"{d}/stats.txt"):
        m = re.match(r'^(\w+)\s+(\d+)\s+(YES|NO)', line.strip())
        if m:
            dur[m.group(1)] = int(m.group(2))
    if not dur:
        sys.exit(f"Error: no per-task rows parsed from {d}/stats.txt")
    per_run.append(dur)
    print(f"# {d.rstrip('/').split('/')[-1]}: {len(dur)} tasks, "
          f"total {sum(dur.values())/3600:.1f}h, "
          f"max/min {max(dur.values())/min(dur.values()):.1f}x", file=sys.stderr)

tasks = sorted(set.intersection(*[set(d) for d in per_run]))
if not tasks:
    sys.exit("Error: the runs share no task names")
print(f"# {len(tasks)} tasks common to all {len(runs)} run(s)", file=sys.stderr)

mean_rank, mean_sec = {}, {}
for t in tasks:
    ranks = []
    for d in per_run:
        ranks.append(sorted(d, key=lambda x: -d[x]).index(t))
    mean_rank[t] = sum(ranks) / len(ranks)
    mean_sec[t] = sum(d[t] for d in per_run) / len(per_run)

order = sorted(tasks, key=lambda t: mean_rank[t])

# Pairwise Spearman: low values mean the order is unstable between these runs and
# a single baked-in list will not help much.
if len(per_run) > 1:
    import itertools
    for i, j in itertools.combinations(range(len(per_run)), 2):
        ri = {t: sorted(per_run[i], key=lambda x: -per_run[i][x]).index(t) for t in tasks}
        rj = {t: sorted(per_run[j], key=lambda x: -per_run[j][x]).index(t) for t in tasks}
        n = len(tasks)
        d2 = sum((ri[t] - rj[t]) ** 2 for t in tasks)
        print(f"# Spearman rho(run{i}, run{j}) = {1 - 6*d2/(n*(n*n-1)):.3f}", file=sys.stderr)

print(f"# longest {order[0]} ~{mean_sec[order[0]]/3600:.1f}h avg, "
      f"shortest {order[-1]} ~{mean_sec[order[-1]]/3600:.1f}h avg", file=sys.stderr)
print("# paste the block below over task_list_all= in "
      "start_robotwin_infer_and_eval.sh", file=sys.stderr)

print("task_list_all=(", end="")
for i, t in enumerate(order):
    if i % 5 == 0:
        print("\n   ", end="")
    print(f' "{t}"', end="")
print("\n)")
PY
