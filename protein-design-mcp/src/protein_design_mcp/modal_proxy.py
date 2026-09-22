"""
Local MCP proxy that forwards tool calls to a Modal GPU deployment.

This runs locally as a standard MCP server (stdio transport) and proxies
all tool calls to the Modal web endpoint via HTTP POST. File arguments
(PDB paths) are read locally and sent inline.

Usage:
    # Set your Modal endpoint URL (printed after `modal deploy`)
    export MODAL_URL=https://<your-workspace>--protein-design-tools.modal.run

    # Start the proxy as an MCP server
    python -m protein_design_mcp.modal_proxy

Configure in Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "protein-design": {
          "command": "python",
          "args": ["-m", "protein_design_mcp.modal_proxy"],
          "env": {
            "MODAL_URL": "https://<your-workspace>--protein-design-tools.modal.run"
          }
        }
      }
    }
"""

import asyncio
import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("protein-design-mcp-modal")

MODAL_URL = os.environ.get("MODAL_URL", "")

# Arguments that reference local PDB/structure files
_FILE_ARGS = {"target_pdb", "complex_pdb", "pdb_path", "expected_structure"}


# Import tool definitions from main server (all 11 tools — GPU available on Modal)
from protein_design_mcp.server import TOOLS  # noqa: E402


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all 11 tools (GPU available on Modal)."""
    return TOOLS


class _NoAutoRedirect303(urllib.request.HTTPRedirectHandler):
    """Stop urllib from silently swallowing Modal's 303 poll redirect.

    Returning None here makes the request fall through to a normal
    HTTPError(303) instead of urllib's default redirect machinery, so
    _request_polling_303 can read the Location header itself.
    """

    def http_error_303(self, req, fp, code, msg, headers):
        return None


_poll_opener = urllib.request.build_opener(_NoAutoRedirect303)


def _request_polling_303(req: "urllib.request.Request", max_polls: int = 40) -> dict:
    """POST to Modal, following its 150s-request-timeout poll pattern.

    Every Modal web endpoint hard-caps a single HTTP request at 150s; a
    slower Function call gets a 303 pointing back at the same URL with a
    tracking query param, and the client is expected to GET that URL
    repeatedly until a non-303 response lands (Modal's docs: up to ~20
    hops / 50 minutes total). Tools like predict_complex (ColabFold) or a
    cold RFdiffusion/ESMFold container regularly exceed 150s, so without
    this the proxy surfaced a bare "Modal endpoint returned 303" error
    instead of the actual result.
    """
    current = req
    for _ in range(max_polls):
        try:
            response = _poll_opener.open(current, timeout=170)
        except urllib.error.HTTPError as e:
            if e.code != 303:
                raise
            location = e.headers.get("Location")
            if not location:
                raise
            current = urllib.request.Request(location, method="GET")
            continue
        return json.loads(response.read())
    raise TimeoutError(f"Modal call did not complete after {max_polls} redirect polls")


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Forward tool call to Modal endpoint."""
    if not MODAL_URL:
        return [
            TextContent(
                type="text",
                text=json.dumps(
                    {
                        "error": (
                            "MODAL_URL not set. Deploy first:\n"
                            "  modal deploy deploy/modal_app.py\n"
                            "Then set MODAL_URL to the printed endpoint URL."
                        )
                    },
                    indent=2,
                ),
            )
        ]

    # Inline local file contents so they travel over HTTP
    prepared = dict(arguments)
    for key in _FILE_ARGS:
        value = prepared.get(key)
        if value and os.path.isfile(value):
            logger.info(f"Inlining local file: {key}={value}")
            with open(value) as f:
                prepared[f"_file_{key}"] = f.read()

    # POST to Modal web endpoint
    payload = json.dumps({"name": name, "arguments": prepared}).encode()
    req = urllib.request.Request(
        MODAL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _request_polling_303(req))
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else str(e)
        result = {"error": f"Modal endpoint returned {e.code}: {body}", "tool": name}
    except urllib.error.URLError as e:
        result = {"error": f"Cannot reach Modal endpoint: {e.reason}", "tool": name}
    except Exception as e:
        result = {"error": f"Proxy error: {e}", "tool": name}

    text = json.dumps(result, indent=2)
    if len(text) > 1_000_000:
        text = json.dumps(result, separators=(",", ":"))

    return [TextContent(type="text", text=text)]


async def run_server():
    """Run the MCP proxy server."""
    logger.info(f"Starting Modal proxy (endpoint: {MODAL_URL or 'NOT SET'})")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main():
    """Main entry point."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
