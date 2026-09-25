#!/usr/bin/env python3
"""Docker HEALTHCHECK for ``van serve``: ``GET /health`` must answer 200.

The port is ``$VAN_HEALTH_PORT``, else the ``--port`` of the serving process (PID 1, read
from ``/proc/1/cmdline``), else ``van serve``'s default port for its ``--protocol``.
Standard library only: the image has no curl.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

DEFAULT_PORTS = {"openai-realtime": 8000, "webrtc": 8080}  # websocket and telephony: 8765


def _option(args: list[str], *names: str) -> str | None:
    """The last value of an option (``--name value`` or ``--name=value``)."""
    value = None
    for i, arg in enumerate(args):
        for name in names:
            if arg == name and i + 1 < len(args):
                value = args[i + 1]
            elif name.startswith("--") and arg.startswith(name + "="):
                value = arg.split("=", 1)[1]
    return value


def serve_port() -> int:
    explicit = os.environ.get("VAN_HEALTH_PORT")
    if explicit:
        return int(explicit)
    try:
        args = Path("/proc/1/cmdline").read_bytes().decode().split("\0")
    except OSError:
        args = []
    given = _option(args, "--port")
    if given and given != "0":
        return int(given)
    protocol = _option(args, "--protocol", "-p") or "openai-realtime"
    return DEFAULT_PORTS.get(protocol, 8765)


def main() -> int:
    url = f"http://127.0.0.1:{serve_port()}/health"
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            return 0 if response.status == 200 else 1
    except OSError as exc:
        print(f"{url}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
