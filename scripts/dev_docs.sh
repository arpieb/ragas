#!/bin/bash
# Serve the docs locally with live reload.
set -euo pipefail

exec uv run --group docs mkdocs serve --dirtyreload
