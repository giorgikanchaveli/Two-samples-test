#!/bin/bash
# =============================================================================
# fetch_results.sh — downloads simulation results from the cluster
# =============================================================================
#
# Run this on your own computer (not on the cluster), after a cluster simulation finishes.
# It replaces copying CSVs by hand with FileZilla:
#   1) downloads the SLURM log files (what the jobs printed) from ~/logs/ on the cluster into cluster_outputs/logs/.
#      Their names contain the job ID, so nothing is overwritten except logs of jobs that are still running.
#   2) downloads every CSV in cluster_outputs/values/ on the cluster,
#   3) puts each one where its plotting script reads it:
#        fp_*.csv, tp_*.csv                                 -> values/fp_tp/
#        mean_*.csv, variance_*.csv, counterexample_*.csv   -> values/comparison_all/
#   4) before replacing anything, moves the old results into old_sim_values/<folder>/<date_time>/.
#
# Which old results get moved:
#   - values/fp_tp/: only a file with the same name (same parameters), since plot_tpfp_csv.jl plots every CSV there.
#   - values/comparison_all/: every file of the same simulation (e.g. all mean_*.csv), since plot_mean.jl etc.
#     plot the first matching file, so there must be only one per simulation.
# A CSV that is identical to the one already in values/ is skipped, so running this twice is harmless.
#
# HOW TO RUN (from anywhere):
#   bash shfiles_cluster/fetch_results.sh             # download and sort
#   bash shfiles_cluster/fetch_results.sh --dry-run   # only print what would be downloaded/moved/copied
#
# Needs passwordless SSH to the cluster under the name `cluster` (a Host entry in ~/.ssh/config), and the
# project cloned into your home folder on the cluster (~/Two-samples-test). See README for the one-time setup.

set -Eeo pipefail                              # stop on error. No -u: macOS bash 3.2 treats empty arrays as unset

REMOTE="cluster:Two-samples-test/cluster_outputs/values/"   # where the cluster jobs save their CSVs
REMOTE_LOGS="cluster:logs/"                    # SLURM log folder (#SBATCH --output/--error in the .sh files)
LOG_DIR="cluster_outputs/logs"
BACKUP_ROOT="old_sim_values"
STAMP=$(date +%Y-%m-%d_%H%M%S)                 # one backup subfolder per run, so older backups are never overwritten

cd "$(dirname "$0")/.."                        # project root, so relative paths work wherever this is run from

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "Dry run: nothing in $LOG_DIR/, values/ or $BACKUP_ROOT/ will change."
fi

# prints the step, and runs it unless this is a dry run
step() {
    echo "  $1"
    shift
    $DRY_RUN || "$@"
}

# folder in values/ that the plotting script for this CSV reads from (empty if unknown)
target_dir() {
    case "$1" in
        fp_*.csv|tp_*.csv)                              echo "values/fp_tp" ;;
        mean_*.csv|variance_*.csv|counterexample_*.csv) echo "values/comparison_all" ;;
        *)                                              echo "" ;;
    esac
}


# --- 1) download log files ---
echo "Downloading logs from $REMOTE_LOGS into $LOG_DIR/ ..."
if $DRY_RUN; then
    rsync -a --dry-run --out-format='  would download %n' "$REMOTE_LOGS" "$LOG_DIR/" | { grep -v '/$' || true; }
else
    mkdir -p "$LOG_DIR"
    rsync -a --out-format='  %n' "$REMOTE_LOGS" "$LOG_DIR/" | { grep -v '/$' || true; }   # prints only new or updated logs
fi


# --- 2) download CSVs into a temporary folder ---
# (not into the local cluster_outputs/values/, which local runs also write to)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Downloading CSVs from $REMOTE ..."
rsync -a --include='*.csv' --exclude='*' "$REMOTE" "$TMP/"
echo "Downloaded $(ls "$TMP" | wc -l | tr -d ' ') CSV file(s)."

# values/comparison_all/ can hold only one CSV per simulation, so stop if the cluster has several
for prefix in mean variance counterexample; do
    matches=( "$TMP/${prefix}"_*.csv )
    if [ -f "${matches[0]}" ] && [ ${#matches[@]} -gt 1 ]; then
        echo "ERROR: the cluster has ${#matches[@]} '${prefix}' CSVs, but values/comparison_all/ can hold only one:"
        for f in "${matches[@]}"; do echo "    $(basename "$f")"; done
        echo "Delete or move the ones you don't want on the cluster, then run this again. Nothing was changed."
        exit 1
    fi
done


# --- 3) and 4) back up old results and copy the new ones ---
for src in "$TMP"/*.csv; do
    [ -f "$src" ] || continue
    name=$(basename "$src")
    dest=$(target_dir "$name")
    if [ -z "$dest" ]; then
        echo "Skipped $name: no plotting script reads it."
        continue
    fi

    # old results this file replaces
    if [ "$dest" = "values/comparison_all" ]; then
        candidates=( "$dest/${name%%_*}"_*.csv ) # every CSV of the same simulation, e.g. mean_*.csv
    else
        candidates=( "$dest/$name" )
    fi
    old=()
    for f in "${candidates[@]}"; do
        [ -f "$f" ] && old+=( "$f" )
    done

    if [ ${#old[@]} -eq 1 ] && [ "${old[0]}" = "$dest/$name" ] && cmp -s "$src" "$dest/$name"; then
        echo "Unchanged: $dest/$name"
        continue
    fi

    echo "$name:"
    backup="$BACKUP_ROOT/$(basename "$dest")/$STAMP"
    for f in "${old[@]}"; do
        $DRY_RUN || mkdir -p "$backup"
        step "back up  $f -> $backup/" mv "$f" "$backup/"
    done
    $DRY_RUN || mkdir -p "$dest"
    step "copy     -> $dest/$name" cp "$src" "$dest/$name"
done

echo "Done."
