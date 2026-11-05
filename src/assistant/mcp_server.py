"""mcp_server.py — expose our tools over the Model Context Protocol (MCP).

This is the same idea as the wallet-agent-mcp you explored earlier: a FastMCP
server whose @mcp.tool() functions can be called by ANY MCP client (Kiro,
Claude Desktop, another agent). The client is the brain; this server is the hands.

Run it:
    uv run python -m assistant.mcp_server          # stdio (for Kiro/desktop clients)
    TRANSPORT=http uv run python -m assistant.mcp_server   # http (for a shared/remote server)
"""

import os
from datetime import datetime

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("learn-ai-agent")


@mcp.tool()
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression like '12 * (3 + 4)'.

    Args:
        expression: a math expression using digits and + - * / ( ) only.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: expression contains disallowed characters."
    try:
        return str(eval(expression))
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


@mcp.tool()
def search_notes(question: str) -> str:
    """Search the user's personal notes about their AWS learning project
    (EC2, IAM roles, security groups, deployment, pgvector). Use for any question
    about the user's own setup. Requires the database to be reachable.

    Args:
        question: what to look up in the notes.
    """
    # Imported lazily so the server still starts (and the other two tools work)
    # even on a machine where the database isn't reachable.
    from assistant.rag import search

    try:
        rows = search(question, k=3)
    except Exception as e:  # noqa: BLE001
        return f"Error searching notes: {e}"
    if not rows:
        return "No matching notes found."
    return "\n\n".join(f"[{source}] {content}" for source, content, _ in rows)


def main() -> None:
    transport = os.environ.get("TRANSPORT", "stdio")
    if transport == "http":
        from mcp.server.transport_security import TransportSecuritySettings

        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("PORT", "8001"))
        # By default MCP only accepts Host: localhost (DNS-rebinding protection).
        # When reached over the network (by IP, or behind a reverse proxy) that
        # check rejects requests with HTTP 421. We disable it here. In production
        # you'd instead run behind an authenticated proxy (like the wallet MCP's nginx).
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
