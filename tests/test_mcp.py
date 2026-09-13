"""Task 14. The lookup tool, over a real MCP round trip."""

import asyncio

import config
from fastmcp import Client

from mcp_server.server import mcp


def test_41_the_tool_answers_for_two_different_record_ids():
    """Spec section 16 test 41, Part 4 criterion 1.

    In-memory transport here, because a test should not bind a port. The real
    client-server round trip over HTTP is scripts/run_part4.py's job and its
    transcript is the evidence for the criterion.
    """
    async def run():
        async with Client(mcp) as client:
            return [
                await client.call_tool(
                    "check_loan_application_status", {"record_id": rid}
                )
                for rid in ("LN-1042", "LN-1057")
            ]

    first, second = asyncio.run(run())
    assert first.data["record_id"] == "LN-1042"
    assert second.data["record_id"] == "LN-1057"
    assert first.data["found"] is True
    assert 0.0 <= first.data["escalation_score"] <= 1.0
    assert first.data != second.data


def test_the_tool_reports_a_missing_record_rather_than_raising():
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(
                "check_loan_application_status", {"record_id": "LN-9999"}
            )

    assert asyncio.run(run()).data["found"] is False


def test_the_tool_carries_the_docstring_the_brief_asks_for():
    async def run():
        async with Client(mcp) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(run())}
    assert "check_loan_application_status" in tools
    description = tools["check_loan_application_status"].description or ""
    assert "escalation" in description.lower()
    assert len(description.split()) >= 20


def test_the_client_never_starts_a_server():
    """D-79. Criterion 14 checks that the client is a separate process."""
    source = (config.REPO_ROOT / "mcp_client.py").read_text(encoding="utf-8")
    for forbidden in ("subprocess", "mcp.run(", "uvicorn", "from mcp_server"):
        assert forbidden not in source, f"mcp_client.py must not reference {forbidden}"
