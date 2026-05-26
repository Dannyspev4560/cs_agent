"""FastMCP server exposing dataset tools for external clients (Task 3).

Run with:
    python -m mcp_server.server

Connect a client:
    from fastmcp import Client
    async with Client("python -m mcp_server.server") as client:
        result = await client.call_tool("count_records", {"category": "REFUND"})
        print(result)
"""

from fastmcp import FastMCP

from agent.tools import (
    count_records,
    get_intent_distribution,
    get_samples,
    list_categories,
    list_intents,
)

mcp = FastMCP("customer-service-analyst")

mcp.tool()(list_categories.func)
mcp.tool()(list_intents.func)
mcp.tool()(count_records.func)
mcp.tool()(get_samples.func)
mcp.tool()(get_intent_distribution.func)

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8000)
