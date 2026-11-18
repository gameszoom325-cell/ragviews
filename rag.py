"""MCP RAG Client Module.

Connects to an external Model Context Protocol (MCP) RAG server,
invokes its retrieval tool for a query, and returns the retrieved
chunks along with a concatenated dummy answer.
"""

import asyncio
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Lazy import MCP client transports so basic imports don't fail without mcp installed
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client
except ImportError:
    ClientSession = None  # type: ignore
    sse_client = None  # type: ignore
    stdio_client = None  # type: ignore
    StdioServerParameters = None  # type: ignore


DEFAULT_TIMEOUT = 30.0  # seconds per query timeout


def get_mcp_config() -> Dict[str, Any]:
    """Retrieve MCP server connection settings from environment variables."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp").strip()
    tool_name = os.getenv("MCP_TOOL_NAME")
    try:
        timeout = float(os.getenv("MCP_TIMEOUT", str(DEFAULT_TIMEOUT)))
    except ValueError:
        timeout = DEFAULT_TIMEOUT

    return {
        "server_url": server_url,
        "tool_name": tool_name.strip() if tool_name else None,
        "timeout": timeout,
    }


def _extract_chunks(call_result: Any) -> List[str]:
    """Extract list of text context chunks from MCP CallToolResult.

    Supports diverse payload shapes including TextContent blocks, JSON-encoded
    lists/dicts, and raw strings.
    """
    chunks: List[str] = []

    content = getattr(call_result, "content", None)
    if content is None:
        if isinstance(call_result, list):
            content = call_result
        elif isinstance(call_result, dict):
            content = [call_result]
        elif isinstance(call_result, str):
            content = [call_result]
        else:
            return []

    for block in content:
        text_val = getattr(block, "text", None)
        if text_val is None and isinstance(block, dict):
            text_val = block.get("text") or block.get("content") or block.get("chunk")
        elif text_val is None and isinstance(block, str):
            text_val = block

        if not text_val:
            continue

        text_str = str(text_val).strip()

        # Check if text is a JSON-encoded array or object of chunks
        if (text_str.startswith("[") and text_str.endswith("]")) or (
            text_str.startswith("{") and text_str.endswith("}")
        ):
            try:
                parsed = json.loads(text_str)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            chunks.append(item.strip())
                        elif isinstance(item, dict):
                            val = item.get("text") or item.get("content") or item.get("chunk")
                            if val:
                                chunks.append(str(val).strip())
                    continue
                elif isinstance(parsed, dict):
                    extracted = (
                        parsed.get("contexts")
                        or parsed.get("chunks")
                        or parsed.get("results")
                        or parsed.get("documents")
                    )
                    if isinstance(extracted, list):
                        for item in extracted:
                            if isinstance(item, str):
                                chunks.append(item.strip())
                            elif isinstance(item, dict):
                                val = item.get("text") or item.get("content") or item.get("chunk")
                                if val:
                                    chunks.append(str(val).strip())
                        continue
            except (json.JSONDecodeError, TypeError):
                pass

        chunks.append(text_str)

    return chunks


async def _resolve_tool_name(session: Any, preferred_tool: Optional[str]) -> str:
    """Determine the target retrieval tool on the MCP server."""
    if preferred_tool:
        return preferred_tool

    try:
        tools_result = await session.list_tools()
        tools = getattr(tools_result, "tools", [])
    except Exception:
        tools = []

    if not tools:
        return "retrieve"

    tool_names = [t.name for t in tools if hasattr(t, "name")]

    candidates = [
        "retrieve",
        "retrieval",
        "search",
        "query",
        "get_contexts",
        "get_context",
        "fetch_contexts",
    ]
    for candidate in candidates:
        if candidate in tool_names:
            return candidate

    for name in tool_names:
        lower = name.lower()
        if "retriev" in lower or "search" in lower or "rag" in lower:
            return name

    return tool_names[0]


async def _call_mcp_query(question: str) -> Dict[str, Any]:
    """Execute asynchronous MCP tool invocation."""
    if ClientSession is None:
        print(
            "Error: 'mcp' Python SDK is not installed.\n"
            "Please install dependencies with: pip install mcp python-dotenv",
            file=sys.stderr,
        )
        sys.exit(1)

    config = get_mcp_config()
    server_target = config["server_url"]
    preferred_tool = config["tool_name"]

    is_http = server_target.startswith("http://") or server_target.startswith("https://")

    try:
        if is_http:
            if sse_client is None:
                raise ImportError("mcp.client.sse is unavailable.")
            async with sse_client(server_target) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_name = await _resolve_tool_name(session, preferred_tool)
                    result = await session.call_tool(
                        name=tool_name,
                        arguments={"query": question, "question": question},
                    )
        else:
            if stdio_client is None or StdioServerParameters is None:
                raise ImportError("mcp.client.stdio is unavailable.")
            parts = shlex.split(server_target)
            if not parts:
                raise ValueError("Empty stdio command.")
            params = StdioServerParameters(command=parts[0], args=parts[1:])
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_name = await _resolve_tool_name(session, preferred_tool)
                    result = await session.call_tool(
                        name=tool_name,
                        arguments={"query": question, "question": question},
                    )

        chunks = _extract_chunks(result)
        top_3 = chunks[:3]
        dummy_answer = "\n\n".join(top_3) if top_3 else ""

        return {
            "answer": dummy_answer,
            "contexts": chunks,
            "response": dummy_answer,
            "retrieved_contexts": chunks,
        }

    except Exception as exc:
        print(
            f"\nError: Could not connect to MCP server at '{server_target}'.\n"
            f"Details: {type(exc).__name__}: {exc}\n\n"
            "Please ensure your MCP RAG server is running (e.g. 'python server.py') "
            "and that MCP_SERVER_URL in your .env is correctly configured.",
            file=sys.stderr,
        )
        sys.exit(1)


def query(question: str) -> Dict[str, Any]:
    """Query the MCP RAG server for retrieved contexts and return a dummy answer.

    Args:
        question: User query string.

    Returns:
        dict: {
            "answer": str,               # Dummy answer: concatenated top 3 chunks
            "contexts": List[str],       # Retrieved context chunk texts
            "response": str,             # Schema alias
            "retrieved_contexts": list,  # Schema alias
        }
    """
    config = get_mcp_config()
    timeout = config["timeout"]

    try:
        return asyncio.run(
            asyncio.wait_for(_call_mcp_query(question), timeout=timeout)
        )
    except asyncio.TimeoutError:
        print(
            f"\nError: MCP server query timed out after {timeout} seconds.\n"
            f"Server target: {config['server_url']}\n"
            f"Question: '{question}'",
            file=sys.stderr,
        )
        sys.exit(1)


# Maintain alias for compatibility
query_rag = query


if __name__ == "__main__":
    sample_question = "What is the primary purpose of the Ragas framework?"
    print(f"Testing MCP query with question: '{sample_question}'")
    out = query(sample_question)
    print("\nResult Answer (Concatenated Top 3 Chunks):")
    print(out["answer"])
    print(f"\nRetrieved {len(out['contexts'])} chunks:")
    for idx, c in enumerate(out["contexts"], 1):
        print(f"[{idx}] {c[:120]}...")
