"""Part 4 Task 14. The lookup tool, exposed over MCP.

One tool, wrapping the same agent/tools.py call the graph uses, so the MCP
surface and the agent cannot answer differently about the same record.

The HTTP transport mounts at config.MCP_PATH, not at the bare root, which is
what the brief is insistent about. The port is config.MCP_PORT rather than
8000, so the Part 3 API keeps its own default. D-79.
"""

from fastmcp import FastMCP

import config
from agent.tools import check_loan_application_status as _lookup

mcp = FastMCP("meridian-loan-support")


@mcp.tool
def check_loan_application_status(record_id: str) -> dict:
    """Look up one Meridian Bank loan application by its record id.

    Returns the application's current status, the sanctioned amount in rupees,
    the age of the application in days, whether it is flagged for fraud
    review, a designed escalation score in [0, 1] combining the fraud flag
    with a normalised staleness signal, and whether that score clears the
    recommended escalation threshold. Record ids look like LN-1042. An id that
    matches no application returns found=False rather than raising, so a
    caller can distinguish a missing record from a failed call.
    """
    return _lookup(record_id)


def main() -> None:
    mcp.run(transport="http", host=config.MCP_HOST, port=config.MCP_PORT)


if __name__ == "__main__":
    main()
