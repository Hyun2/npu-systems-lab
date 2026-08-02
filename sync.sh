#!/usr/bin/env bash
# Sync this repo to the GPU machine and back.
#
#   ./sync.sh check          show what differs (dry run, changes nothing)
#   ./sync.sh push           local -> server, source only
#   ./sync.sh pull-results   server -> local, results only (for committing)
#
# Division of labour:
#   rsync = transport. Push uncommitted work to the machine that can run it.
#   git   = history. Commit at meaningful boundaries, not to move bytes.
#
# Rule: never edit files on the server. `push` runs with --delete and will
# overwrite server-side edits. If you must, run `check` first to see them.

set -euo pipefail

# Override for your own machine:  NPU_HOST=myserver ./sync.sh push
REMOTE_HOST="${NPU_HOST:-i-u}"
REMOTE_DIR="${NPU_DIR:-~/dev/npu-systems-lab}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Excluded from push: things that belong only on the server (envs, weights,
# build output) or only on the client (git metadata, local-only helpers).
EXCLUDES=(
  --exclude '.git/'
  --exclude '.venv/'
  --exclude 'venv/'
  --exclude '__pycache__/'
  --exclude 'models/'
  --exclude 'results/'
  --exclude '_refs'
  --exclude 'CLAUDE.local.md'
  --exclude '.serena/'
  --exclude '.DS_Store'
  --exclude '*.gguf'
  --exclude '*.safetensors'
  --exclude '*.onnx*'
)

case "${1:-check}" in
  check)
    echo "Comparing ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_DIR}"
    echo "(dry run -- nothing is modified. Empty output means in sync.)"
    # --checksum, and no -p/-t: compare file CONTENT only.
    # git rewrites mtimes on clone and pull, so a timestamp-based comparison
    # cries wolf after every git operation and stops being trustworthy.
    rsync -rlzc --delete --dry-run --itemize-changes "${EXCLUDES[@]}" \
      "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/" \
      | grep -v '^\.d\.\.t\.\.\.\.\.$' || true
    ;;

  push)
    rsync -az --delete --itemize-changes "${EXCLUDES[@]}" \
      "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"
    echo "pushed to ${REMOTE_HOST}:${REMOTE_DIR}"
    ;;

  pull-results)
    # Results are generated on the server and committed from here.
    # Raw tensor dumps stay behind -- see .gitignore.
    # Errors are NOT suppressed. A missing remote path means the server has
    # not run this project yet, or a directory was renamed here but not there.
    # Reporting success either way is worse than failing: the caller commits
    # whatever is already on disk and believes it came from the server.
    failed=0
    for d in "${LOCAL_DIR}"/[0-9][0-9]-*/; do
      proj="$(basename "$d")"
      mkdir -p "${d}results"
      if ! rsync -az --itemize-changes --exclude 'raw/' \
        "${REMOTE_HOST}:${REMOTE_DIR}/${proj}/results/" "${d}results/"; then
        echo "FAILED ${proj} -- is ${REMOTE_DIR}/${proj}/results/ there?" >&2
        failed=1
      fi
    done
    [ "$failed" -eq 0 ] || { echo "pull-results incomplete" >&2; exit 1; }
    echo "pulled results into ${LOCAL_DIR}"
    ;;

  *)
    sed -n '2,10p' "${BASH_SOURCE[0]}"
    exit 1
    ;;
esac
