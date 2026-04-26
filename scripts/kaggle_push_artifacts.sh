#!/bin/bash
set -euo pipefail

# Required env vars:
# - GITHUB_TOKEN
# - GIT_USER_NAME
# - GIT_USER_EMAIL
#
# Optional env vars:
# - TARGET_BRANCH (default: main)
# - REMOTE_NAME (default: origin)

TARGET_BRANCH="${TARGET_BRANCH:-main}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

if [[ -z "${GITHUB_TOKEN:-}" || -z "${GIT_USER_NAME:-}" || -z "${GIT_USER_EMAIL:-}" ]]; then
  echo "Missing required env vars. Need: GITHUB_TOKEN, GIT_USER_NAME, GIT_USER_EMAIL"
  exit 1
fi

if [[ ! -f "artifacts/LATEST.txt" ]]; then
  echo "artifacts/LATEST.txt not found. Run package step first."
  exit 1
fi

LATEST_RUN="$(tr -d '\r\n' < artifacts/LATEST.txt)"
if [[ -z "${LATEST_RUN}" ]]; then
  echo "LATEST run is empty."
  exit 1
fi

echo "Preparing commit for artifacts/${LATEST_RUN} ..."
git add "artifacts/${LATEST_RUN}" "artifacts/LATEST.txt"

if git diff --staged --quiet; then
  echo "No new artifact changes to commit."
  exit 0
fi

git -c user.name="${GIT_USER_NAME}" -c user.email="${GIT_USER_EMAIL}" commit -m "Add Kaggle artifact ${LATEST_RUN}"

REPO_URL="$(git remote get-url "${REMOTE_NAME}")"

if [[ "${REPO_URL}" == https://github.com/* ]]; then
  AUTH_URL="${REPO_URL/https:\/\//https:\/\/${GITHUB_TOKEN}@}"
  git push "${AUTH_URL}" "HEAD:${TARGET_BRANCH}"
else
  echo "Remote URL is not HTTPS GitHub URL."
  echo "Push manually after setting an authenticated remote."
  exit 1
fi

echo "✅ Artifact pushed to ${TARGET_BRANCH}"
