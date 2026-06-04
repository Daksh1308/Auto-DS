#!/usr/bin/env bash
# scripts/set-repo-meta.sh
#
# Sets the GitHub repo description and topics via the REST API.
#
# Usage:
#   GITHUB_TOKEN=<your PAT with repo scope> \
#   ./scripts/set-repo-meta.sh <owner> <repo>
#
# Example:
#   GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx \
#   ./scripts/set-repo-meta.sh Daksh1308 Auto-DS
#
# Get a PAT at: https://github.com/settings/tokens  (scope: repo)

set -euo pipefail

OWNER="${1:-Daksh1308}"
REPO="${2:-Auto-DS}"

DESCRIPTION="Multi-phase data automation tool: upload CSV/XLSX, get cleaned data, insights, Plotly dashboard, AI chat with pandas sandbox, and one-click AutoML-lite."

TOPICS='[
  "data-science",
  "data-cleaning",
  "fastapi",
  "react",
  "typescript",
  "vite",
  "plotly",
  "pandas",
  "scikit-learn",
  "automl",
  "dashboard",
  "ai-chat",
  "llm",
  "data-visualization",
  "python",
  "csv"
]'

: "${GITHUB_TOKEN:?Set GITHUB_TOKEN to a PAT with the 'repo' scope}"

echo "Setting description on $OWNER/$REPO ..."
curl -sS -X PATCH \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${OWNER}/${REPO}" \
  -d "$(jq -n --arg desc "$DESCRIPTION" '{description: $desc}')"

echo
echo "Setting topics on $OWNER/$REPO ..."
curl -sS -X PUT \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/${OWNER}/${REPO}/topics" \
  -d "$(jq -n --argjson t "$TOPICS" '{names: $t}')"

echo
echo "Done. Verify at: https://github.com/${OWNER}/${REPO}"
