#!/usr/bin/env bash
# PostToolUse hook: format and lint any Python file Claude just edited or wrote.
# Receives the hook payload as JSON on stdin; needs jq and ruff on PATH.
set -euo pipefail

FILE_PATH=$(jq -r '.tool_input.file_path // empty')

if [[ -n "$FILE_PATH" && "$FILE_PATH" == *.py && -f "$FILE_PATH" ]]; then
  ruff format --quiet "$FILE_PATH" || true
  # Show remaining lint problems to Claude so it can fix them in the next step.
  ruff check --fix --quiet "$FILE_PATH" || true
fi

exit 0
