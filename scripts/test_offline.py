"""Run regression tests with outbound connections blocked (no paid API calls)."""
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


def deny_network(*args, **kwargs):
    raise RuntimeError("Network disabled during offline regression tests")


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    with patch.object(socket.socket, "connect", deny_network), \
         patch.object(socket.socket, "connect_ex", deny_network), \
         patch.object(socket, "create_connection", deny_network):
        suite = unittest.defaultTestLoader.discover(str(root / "tests"))
        result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
