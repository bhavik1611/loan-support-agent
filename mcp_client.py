"""Part 4 Task 14. A standalone MCP client.

A different file and a different process from the agent, which is what the
criterion checks. It takes a URL and never starts a server: a client that
launches its own server is no longer only a client. D-79.

Run it against a server already listening:

    .venv/bin/python mcp_client.py --url http://127.0.0.1:8765/mcp LN-1042 LN-1057
"""

import argparse
import asyncio
import json

from fastmcp import Client

TOOL = "check_loan_application_status"


async def call(url: str, record_ids: list[str]) -> list[dict]:
    """One session, one call per record id, in order."""
    results = []
    async with Client(url) as client:
        for record_id in record_ids:
            response = await client.call_tool(TOOL, {"record_id": record_id})
            results.append({
                "record_id": record_id,
                "is_error": response.is_error,
                "content": [block.text for block in response.content],
                "structured": response.data,
            })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the Meridian MCP tool.")
    parser.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8765/mcp")
    parser.add_argument("record_ids", nargs="+", help="two or more record ids")
    args = parser.parse_args()

    for result in asyncio.run(call(args.url, args.record_ids)):
        print(f"--- {TOOL}({result['record_id']!r}) ---")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
