#!/usr/bin/env bash
set -euo pipefail

DIFY_BASE_URL="${DIFY_BASE_URL:-http://127.0.0.1:8081}"
echo "Checking Dify reachability at ${DIFY_BASE_URL}"
curl --fail --silent --show-error --max-time 10 "${DIFY_BASE_URL}" >/dev/null
echo "Dify endpoint is reachable. Use scripts/run_demo.py --mode dify after configuring DIFY_APP_API_KEY."
