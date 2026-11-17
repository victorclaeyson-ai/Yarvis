"""A tiny MCP server for the tests: started by Yarvis as a real program (stdio)."""

import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

server = MCPServer("notes")


@server.tool()
def add_note(text: str) -> str:
    """Store a note."""
    return f"Stored: {text}"


@server.tool()
def environment() -> str:
    """Which variables this server was given (names only), and where it runs."""
    return f"{' '.join(sorted(os.environ))}|{Path.cwd()}"


if __name__ == "__main__":
    sys.stderr.write("notes server starting\n")
    server.run()
