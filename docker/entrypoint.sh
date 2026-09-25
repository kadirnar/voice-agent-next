#!/bin/sh
# Container entrypoint: `van serve` bound to all interfaces by default.
#
#   docker run IMAGE                         -> van serve --host 0.0.0.0 --protocol websocket
#   docker run IMAGE --preset local-cpu ...  -> van serve --host 0.0.0.0 --preset local-cpu ...
#   docker run IMAGE serve -p webrtc         -> van serve --host 0.0.0.0 -p webrtc
#   docker run IMAGE providers               -> van providers (any van command)
#   docker run IMAGE python -c ...           -> python -c ... (any program on PATH)
#
# A `--host` given by the caller wins: the last occurrence of an option is used.
set -e

case "${1:-}" in
    "" | -*)
        exec van serve --host "${VAN_HOST:-0.0.0.0}" "$@"
        ;;
    serve)
        shift
        exec van serve --host "${VAN_HOST:-0.0.0.0}" "$@"
        ;;
esac

if command -v "$1" >/dev/null 2>&1; then
    exec "$@"
fi
exec van "$@"
