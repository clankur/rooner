#!/usr/bin/env bash
# Prepare the pinned live runtime without replacing an existing SDK checkout.
set -euo pipefail
BRIDGE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SDK_DIR="$BRIDGE_DIR/rs-sdk"
SDK_REVISION=66ed33320e719e92883d5fbfc05a0b091b54373f

if [ "$(bun --version)" != "1.3.14" ]; then
    echo "RLVR requires Bun 1.3.14" >&2
    exit 1
fi
if [ ! -d "$SDK_DIR" ]; then
    git clone --no-checkout https://github.com/MaxBittker/rs-sdk.git "$SDK_DIR"
    git -C "$SDK_DIR" checkout --detach "$SDK_REVISION"
elif [ "$(git -C "$SDK_DIR" rev-parse HEAD)" != "$SDK_REVISION" ]; then
    echo "Existing SDK is not at the pinned revision; preserve it and prepare a separate checkout." >&2
    exit 1
fi

(cd "$BRIDGE_DIR" && bun install --frozen-lockfile)
for component in "" server/engine server/gateway server/webclient; do
    (cd "$SDK_DIR/$component" && bun install --frozen-lockfile)
done
(cd "$SDK_DIR/server/engine" && bun run build)
(cd "$SDK_DIR/server/webclient" && bun run build:dev)
(cd "$BRIDGE_DIR" && bun x puppeteer browsers install chrome)
echo "Runtime prepared. Run uv run python -m rsenv.rlvr doctor from the repository root, then a scripted episode."
