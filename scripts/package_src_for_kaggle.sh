#!/usr/bin/env bash
# Upload the team's shared code (src/) to Kaggle as the private dataset "casmi-team-src".
# Teammates add it to their submission notebook as an input (internet is off during submission).
# Run from the repo root. First time: creates the dataset. Later: pushes a new version.
set -euo pipefail
USER=$(kaggle config view | awk '/username/ {print $3}')
STAGE=kaggle_upload/casmi-team-src
rm -rf "$STAGE" && mkdir -p "$STAGE/casmi_src"
rsync -a --exclude '__pycache__' src "$STAGE/casmi_src/"
cat > "$STAGE/dataset-metadata.json" <<JSON
{"title": "casmi-team-src", "id": "$USER/casmi-team-src", "licenses": [{"name": "MIT"}]}
JSON
if kaggle datasets status "$USER/casmi-team-src" >/dev/null 2>&1; then
  kaggle datasets version -p "$STAGE" -m "update src $(git rev-parse --short HEAD)" -r zip
else
  kaggle datasets create -p "$STAGE" -r zip
fi
echo "Done. Share it with teammates on Kaggle: Dataset page → Settings → Sharing."
