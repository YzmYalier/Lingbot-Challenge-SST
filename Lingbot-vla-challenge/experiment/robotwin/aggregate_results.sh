#!/bin/bash
# Aggregate a RoboTwin eval run into one benchmark score.
#
# Works for both single-node and multi-node runs: the source of truth is
# <run_dir>/eval_logs/<task>.log, one per task, written by whichever node claimed
# it. Per-node stats_n*.txt only cover that node's share, so they must not be
# read as a score.
#
# Usage: bash aggregate_results.sh <run_dir> [test_num]
#   test_num: episodes per task a complete task must have. Default: what the
#   launcher recorded in <run_dir>/test_num, else 100 (runs predating that file).
set -uo pipefail

run_dir="${1:-}"
if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
    echo "Usage: bash $0 <run_dir> [test_num]" >&2
    exit 2
fi
log_dir="${run_dir}/eval_logs"
if [ ! -d "$log_dir" ]; then
    echo "Error: ${log_dir} not found (is ${run_dir} a run directory?)" >&2
    exit 2
fi

test_num="${2:-}"
if [ -z "$test_num" ] && [ -f "${run_dir}/test_num" ]; then
    test_num="$(tr -d '[:space:]' < "${run_dir}/test_num")"
fi
test_num="${test_num:-100}"
if ! [[ "$test_num" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: test_num must be a positive integer, got '${test_num}'" >&2
    exit 2
fi
echo "Episodes per task  : ${test_num}"

printf '%-32s %-14s %-10s\n' "Task" "Success/Total" "Rate"
echo "--------------------------------------------------------------"

total_success=0
total_episodes=0
tasks_seen=0
incomplete=()

for log in "$log_dir"/*.log; do
    [ -e "$log" ] || continue
    task="$(basename "$log" .log)"
    # Last "Success rate: S/T => R%" line wins; strip ANSI colour first.
    last="$(sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -oP 'Success rate: \K.*' | tail -1)"
    if [ -z "$last" ]; then
        printf '%-32s %-14s %-10s\n' "$task" "-" "no result"
        incomplete+=("${task}(0/${test_num})")
        tasks_seen=$((tasks_seen + 1))
        continue
    fi
    suc="$(echo "$last"  | grep -oP '^\d+' | head -1)"
    tot="$(echo "$last"  | grep -oP '/\K\d+' | head -1)"
    rate="$(echo "$last" | grep -oP '=> \K[\d.]+')"
    if [ -z "$suc" ] || [ -z "$tot" ]; then
        printf '%-32s %-14s %-10s\n' "$task" "-" "unparsed"
        incomplete+=("${task}(parse)")
        tasks_seen=$((tasks_seen + 1))
        continue
    fi
    printf '%-32s %-14s %-10s\n' "$task" "${suc}/${tot}" "${rate}%"
    total_success=$((total_success + suc))
    total_episodes=$((total_episodes + tot))
    tasks_seen=$((tasks_seen + 1))
    [ "$tot" -ne "$test_num" ] && incomplete+=("${task}(${tot}/${test_num})")
done

echo "--------------------------------------------------------------"
if [ "$total_episodes" -gt 0 ]; then
    overall="$(awk "BEGIN{printf \"%.2f\", ${total_success}/${total_episodes}*100}")"
else
    overall="0.00"
fi
echo "Tasks with results : ${tasks_seen}"
echo "Episodes           : ${total_episodes}"
echo "Successes          : ${total_success}"
echo "Overall rate       : ${overall}%   (micro-average over all episodes)"

# Claim bookkeeping: a task claimed but with no result means its owner died.
if [ -d "${run_dir}/claims" ]; then
    claimed="$(ls -1 "${run_dir}/claims" 2>/dev/null | wc -l)"
    echo "Tasks claimed      : ${claimed}"
fi
if [ ${#incomplete[@]} -gt 0 ]; then
    echo ""
    echo "Warning: ${#incomplete[@]} task(s) did not reach ${test_num} episodes:"
    printf '  %s\n' "${incomplete[@]}"
    echo "  A claimed task with no result means the node that owned it died;"
    echo "  rm -rf ${run_dir}/claims/<task> and rerun to let a node re-claim it."
fi
