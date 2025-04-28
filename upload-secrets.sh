#!/usr/bin/env bash
set -euo pipefail

# -- Default configurations --
ENV_FILE=".env.dev"
PREFIX="DEV_"
SUFFIX=""

# -- Help/Usage function --
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --env-file FILE   Specify a custom .env file. Default: .env
  --prefix STR      Specify a prefix to prepend to each secret name.
  --suffix STR      Specify a suffix to append to each secret name.
  --help            Show this help message.

Examples:
  $(basename "$0") --prefix DEV_
  $(basename "$0") --env-file .env.production --suffix _PROD

Description:
  Reads a .env file line by line, skipping comments and empty lines,
  and uploads each variable as a GitHub secret using the gh CLI.
  Optionally adds a prefix and/or suffix to the secret name.
EOF
}

# -- Parse Command-Line Arguments --
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --prefix)
      PREFIX="$2"
      shift 2
      ;;
    --suffix)
      SUFFIX="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# -- Verify .env file exists --
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: .env file '$ENV_FILE' not found."
  exit 1
fi

echo "Using env file: $ENV_FILE"
echo "Using prefix: '$PREFIX'"
echo "Using suffix: '$SUFFIX'"
echo

# -- Read .env file line by line --
while IFS= read -r line || [[ -n "$line" ]]; do
  # 1. Skip empty lines or commented lines
  if [[ -z "$line" ]] || [[ "$line" =~ ^# ]]; then
    continue
  fi

  # 2. Extract KEY and VALUE
  KEY=$(echo "$line" | cut -d '=' -f 1)
  VALUE=$(echo "$line" | cut -d '=' -f 2-)

  # Trim whitespace (leading/trailing)
  KEY=$(echo "$KEY" | xargs)
  VALUE=$(echo "$VALUE" | xargs)

  # 3. If value is empty, skip or handle as needed
  if [[ -z "$VALUE" ]]; then
    echo "Skipping '$KEY' because the value is empty."
    continue
  fi

  # 4. Construct final secret name using prefix/suffix
  FINAL_KEY="${PREFIX}${KEY}${SUFFIX}"

  # 5. Upload the secret using GitHub CLI
  echo "Uploading secret: $FINAL_KEY"
  gh secret set "$FINAL_KEY" -b "$VALUE"

  # Optional: Print out the secret value that was set (NOT recommended for production!)
   echo "Secret '$FINAL_KEY' was set with value: '$VALUE'"

done < "$ENV_FILE"

echo
echo "All done!"
