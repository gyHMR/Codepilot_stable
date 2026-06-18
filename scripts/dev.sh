#!/usr/bin/env bash
# Codepilot Docker development launcher.
#
# Usage:
#   ./scripts/dev.sh
#   ./scripts/dev.sh --mode cli
#   ./scripts/dev.sh --mode im --transport longconn

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-im}"
export CODEPILOT_TRANSPORT="${CODEPILOT_TRANSPORT:-webhook}"
export CODEPILOT_HOST="${CODEPILOT_HOST:-0.0.0.0}"
export CODEPILOT_PORT="${CODEPILOT_PORT:-8787}"
export CODEPILOT_WORKSPACE="${CODEPILOT_WORKSPACE:-/workspace}"
export CODEPILOT_LOG_LEVEL="${CODEPILOT_LOG_LEVEL:-debug}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)       MODE="$2"; export MODE; shift 2 ;;
        --transport)  CODEPILOT_TRANSPORT="$2"; export CODEPILOT_TRANSPORT; shift 2 ;;
        --host)       CODEPILOT_HOST="$2"; export CODEPILOT_HOST; shift 2 ;;
        --port)       CODEPILOT_PORT="$2"; export CODEPILOT_PORT; shift 2 ;;
        --workspace)  CODEPILOT_WORKSPACE="$2"; export CODEPILOT_WORKSPACE; shift 2 ;;
        --log-level)  CODEPILOT_LOG_LEVEL="$2"; export CODEPILOT_LOG_LEVEL; shift 2 ;;
        *)            echo "[dev] Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$MODE" == "im" ]]; then
    echo "[dev] Building and starting Codepilot IM service with Docker Compose ..."
    docker compose up --build codepilot-im
elif [[ "$MODE" == "cli" ]]; then
    echo "[dev] Building and starting Codepilot CLI with Docker Compose ..."
    docker compose build codepilot-cli
    docker compose run --rm codepilot-cli
else
    echo "[dev] Unknown mode: $MODE"
    exit 1
fi
