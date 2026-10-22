"""`python -m localhydra [--config FILE] [--host IP] [--port N]` - loopback-only server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn
from pydantic import ValidationError

from .app import create_app
from .contracts.config import ServerConfig
from .settings import ConfigError, Settings, load_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="localhydra")
    ap.add_argument("--config", type=Path, default=os.environ.get("LOCALHYDRA_CONFIG"))
    ap.add_argument("--host", default=None, help="loopback IP literal (default 127.0.0.1)")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args(argv)
    try:
        settings = load_settings(args.config)
        if args.host is not None or args.port is not None:
            server = ServerConfig(
                host=args.host or settings.server.host, port=args.port or settings.server.port
            )
            settings = Settings(**{**settings.__dict__, "server": server})
    except (ConfigError, ValidationError) as e:
        print(f"localhydra: configuration error: {e}", file=sys.stderr)
        return 2
    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_level="info",
        server_header=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
