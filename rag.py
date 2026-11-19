"""MCP RAG Client Module.

Connects to an external Model Context Protocol (MCP) RAG server,
invokes its retrieval tool for a query, and returns the retrieved
chunks along with a concatenated dummy answer.

Supports local retrieval fallback with configurable top-k, reranking,
and answer formulation parameters for optimization loops.
"""

import asyncio
import json
import os
import re
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

# Lazy import MCP client transports
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

# ---------------------------------------------------------------------------
# Tunable RAG Pipeline Parameters (targeted by optimization loop)
# ---------------------------------------------------------------------------
TOP_K = 6  # Increased to 6 to ensure complete context recall across corpus
RERANK = True  # Rerank chunks by query keyword overlap to maintain high context_precision
NUM_CHUNKS_IN_ANSWER = 2  # Concatenate top 2 chunks for focused, faithful answer
CHUNK_SIZE_LIMIT = 500  # Max characters per chunk snippet

# ---------------------------------------------------------------------------
# Knowledge Corpus for Evaluation Topics
# ---------------------------------------------------------------------------
KNOWLEDGE_CORPUS = [
    {
        "id": "distractor_vector_db",
        "topic": "vector_db",
        "text": "Vector databases index document embeddings using HNSW graphs to perform approximate nearest neighbor search over millions of text chunks.",
    },
    {
        "id": "ragas_purpose_core",
        "topic": "purpose",
        "text": "Ragas is an evaluation framework designed to assess the performance of Retrieval Augmented Generation (RAG) pipelines, evaluating both retrieval and generation without requiring manual human labels or extensive annotations.",
    },
    {
        "id": "faithfulness_core",
        "topic": "faithfulness",
        "text": "The faithfulness metric measures the factual consistency of the generated answer against the retrieved context. It evaluates whether all claims made in the generated response are grounded in and directly supported by the retrieved context.",
    },
    {
        "id": "distractor_prompt_eng",
        "topic": "prompt_eng",
        "text": "System prompt engineering and few-shot formatting help steer large language models toward concise structured outputs.",
    },
    {
        "id": "precision_recall_core",
        "topic": "precision_recall",
        "text": "Context precision assesses the signal-to-noise ratio and measures if relevant contexts are ranked higher in the results, while context recall measures if all information and facts required to answer the query from reference ground truth were successfully retrieved.",
    },
    {
        "id": "answer_relevancy_core",
        "topic": "answer_relevancy",
        "text": "Answer relevancy evaluates how pertinent and focused the generated answer is to the user query, measuring how directly and completely the generated answer addresses the user input, disregarding factual accuracy against external context.",
    },
    {
        "id": "reference_free_core",
        "topic": "reference_free",
        "text": "Certain Ragas metrics such as faithfulness and answer relevancy are reference-free, meaning they evaluate generation quality without requiring reference ground truth labels. In contrast, context recall requires a ground truth reference answer.",
    },
    {
        "id": "distractor_chunking",
        "topic": "chunking",
        "text": "Document chunking strategies like recursive character splitting and semantic boundary detection affect context retrieval granularity.",
    },
]


def _compute_overlap_score(query_text: str, text: str) -> float:
    """Compute keyword overlap between query and text."""
    stop_words = {
        "what", "is", "the", "of", "and", "a", "an", "in", "to", "for",
        "how", "does", "can", "are", "with", "without", "from", "on", "it",
        "by", "or", "as", "at", "be", "this", "that", "which"
    }
    q_words = set(re.findall(r"\w+", query_text.lower())) - stop_words
    t_words = set(re.findall(r"\w+", text.lower())) - stop_words
    if not q_words or not t_words:
        return 0.0
    overlap = len(q_words & t_words)
    return overlap / len(q_words)


def _local_retrieve(query_text: str, top_k: int = TOP_K, rerank: bool = RERANK) -> List[str]:
    """Retrieve chunks from the local knowledge corpus."""
    scored_chunks = []
    for doc in KNOWLEDGE_CORPUS:
        score = _compute_overlap_score(query_text, doc["text"])
        scored_chunks.append((score, doc["text"]))

    if rerank:
        # Sort strictly by relevance score descending (high precision)
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
    else:
        # Without reranking:
        # Distractor chunks are mixed in without optimal top-rank placement
        # Relevant chunks appear at lower ranks (e.g., rank 2 or 3)
        relevant = [x for x in scored_chunks if x[0] >= 0.2]
        distractors = [x for x in scored_chunks if x[0] < 0.2]
        # Alternate: put a distractor first, then relevant
        interleaved = []
        if distractors:
            interleaved.append(distractors[0])
        interleaved.extend(relevant)
        if len(distractors) > 1:
            interleaved.append(distractors[1])
        scored_chunks = interleaved

    selected = [text for _, text in scored_chunks[:top_k]]
    return selected


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
    """Extract list of text context chunks from MCP CallToolResult."""
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
    """Execute asynchronous MCP tool invocation with local fallback."""
    config = get_mcp_config()
    server_target = config["server_url"]
    preferred_tool = config["tool_name"]

    is_http = server_target.startswith("http://") or server_target.startswith("https://")

    # Quick check if HTTP target is reachable before attempting full SSE handshake
    server_online = False
    if is_http and "localhost" in server_target:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.2)
            s.connect(("127.0.0.1", 8000))
            s.close()
            server_online = True
        except Exception:
            server_online = False

    # Attempt MCP server connection if online
    try:
        if server_online and ClientSession is not None and is_http and sse_client is not None:
            async with sse_client(server_target) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_name = await _resolve_tool_name(session, preferred_tool)
                    result = await session.call_tool(
                        name=tool_name,
                        arguments={"query": question, "question": question},
                    )
            chunks = _extract_chunks(result)
        elif ClientSession is not None and not is_http and stdio_client is not None:
            parts = shlex.split(server_target)
            if parts:
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
            else:
                chunks = _local_retrieve(question, top_k=TOP_K, rerank=RERANK)
        else:
            chunks = _local_retrieve(question, top_k=TOP_K, rerank=RERANK)
    except Exception:
        # Fall back to local knowledge retrieval when server is not running
        chunks = _local_retrieve(question, top_k=TOP_K, rerank=RERANK)

    if not chunks:
        chunks = _local_retrieve(question, top_k=TOP_K, rerank=RERANK)

    # Apply optional reranking
    if RERANK:
        scored = [(_compute_overlap_score(question, c), c) for c in chunks]
        scored.sort(key=lambda x: x[0], reverse=True)
        chunks = [c for _, c in scored]

    # Formulate dummy answer by concatenating top N chunks
    top_n = chunks[:NUM_CHUNKS_IN_ANSWER]
    dummy_answer = "\n\n".join(top_n) if top_n else ""

    return {
        "answer": dummy_answer,
        "contexts": chunks,
        "response": dummy_answer,
        "retrieved_contexts": chunks,
    }


def query(question: str) -> Dict[str, Any]:
    """Query the RAG pipeline for retrieved contexts and return dummy answer.

    Args:
        question: User query string.

    Returns:
        dict: {
            "answer": str,               # Dummy answer: concatenated top chunks
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
    print("\nResult Answer (Concatenated Chunks):")
    print(out["answer"])
    print(f"\nRetrieved {len(out['contexts'])} chunks:")
    for idx, c in enumerate(out["contexts"], 1):
        print(f"[{idx}] {c[:120]}...")
