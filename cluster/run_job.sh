#!/bin/bash
# One HTCondor job: run a slice of the scenario set under one model variant.
#
# Arguments:  $1 = model (full | uniform | myopic | baseline)
#             $2 = optional scenario limit (for smoke tests)
#
# Results are written under $RSA_PLANNING_OUTPUT so that they land on scratch
# rather than in the repository. Runs are resumable: a scenario whose result
# file already exists is skipped, so a re-submitted job picks up where it left
# off, and several jobs can share one output directory.
set -euo pipefail

MODEL="${1:-full}"
LIMIT="${2:-}"

REPO_DIR="${REPO_DIR:-$HOME/rsa_planning_aamas26}"
export RSA_PLANNING_OUTPUT="${RSA_PLANNING_OUTPUT:-/scratch/$USER/rsa_planning_output}"
mkdir -p "$RSA_PLANNING_OUTPUT"

echo "Node:   $HOSTNAME"
echo "Model:  $MODEL"
echo "Output: $RSA_PLANNING_OUTPUT"
echo "Date:   $(date)"
python3 --version

cd "$REPO_DIR"

# Build the sequence tree once; later jobs reuse it.
python3 run_pipeline.py precompute --horizon 7

if [ -n "$LIMIT" ]; then
    python3 run_pipeline.py simulate --model "$MODEL" --limit "$LIMIT" --processes "${OMP_NUM_THREADS:-30}"
else
    python3 run_pipeline.py simulate --model "$MODEL" --processes "${OMP_NUM_THREADS:-30}"
fi

echo "--- finished $MODEL ---"
