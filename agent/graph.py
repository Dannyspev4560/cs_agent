import json
import os
import re
from typing import Annotated

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from langchain_core.runnables import RunnableConfig

from agent.profile import load_profile, save_profile
from agent.prompts import AGENT_PROMPT, PROFILE_UPDATE_PROMPT, ROUTER_PROMPT
from agent.tools import ALL_TOOLS

load_dotenv()

MAX_ITERATIONS = 12

# 70B for the agent loop — tool selection and multi-step reasoning benefit from scale
_llm = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("Nebius_API_key"),
    model="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0,
)

# 27B for classification/extraction tasks — router and profile update only need
# structured JSON output, not full reasoning capacity
_llm_small = ChatOpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("Nebius_API_key"),
    model="google/gemma-3-27b-it",
    temperature=0,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str       # set by router: "structured" | "unstructured" | "out_of_scope"
    iteration_count: int  # incremented each agent loop to enforce MAX_ITERATIONS


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# todo: check later with smaller LLM model 
def router(state: AgentState) -> dict:
    """Classify the latest user message and store query_type in state."""
    last_message = state["messages"][-1]

    response = _llm_small.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=str(last_message.content)),
    ])

    # Parse the JSON the router prompt guarantees
    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    try:
        query_type = json.loads(match.group())["query_type"]
        if query_type not in {"structured", "unstructured", "out_of_scope"}:
            query_type = "out_of_scope"
    except Exception:
        query_type = "out_of_scope"

    print(f"\n[Router] → {query_type}")
    return {"query_type": query_type, "iteration_count": 0}


def agent(state: AgentState, config: RunnableConfig) -> dict:
    """Main ReAct node: call the LLM with tools bound."""
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return {"messages": [AIMessage(content=(
            "I've reached my maximum reasoning steps without a final answer. "
            "Please try rephrasing or simplifying your question."
        ))]}

    # Print the observation the agent is about to reason over
    last = state["messages"][-1]
    if last.type == "tool":
        print(f"  [Observation] {last.name}: {str(last.content)[:300]}")
    elif last.type == "ai" and hasattr(last, "tool_calls") and last.tool_calls:
        for tc in last.tool_calls:
            print(f"  [Tool call]   {tc['name']}({tc['args']})")

    session_id = config.get("configurable", {}).get("thread_id", "default")
    profile = load_profile(session_id)
    system = SystemMessage(
        content=AGENT_PROMPT.format(
            query_type=state.get("query_type", "structured"),
            user_profile=json.dumps(profile, indent=2),
        )
    )
    # Prepend system message if not already present
    messages = state["messages"]
    if not messages or messages[0].type != "system":
        messages = [system] + list(messages)

    response = _llm.bind_tools(ALL_TOOLS).invoke(messages)

    return {
        "messages": [response],
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


def decline(state: AgentState) -> dict:
    """Politely refuse out-of-scope queries."""
    return {"messages": [AIMessage(content=(
        "I'm only able to answer questions about the Bitext Customer Service dataset. "
        "Your question appears to be outside that scope — I can't help with it."
    ))]}


def update_profile(state: AgentState, config: RunnableConfig) -> dict:
    """Extract facts from the current turn and persist them to the user profile."""
    session_id = config.get("configurable", {}).get("thread_id", "default")
    profile = load_profile(session_id)

    # Use the last 6 messages for context (skip system messages)
    recent = [m for m in state["messages"][-6:] if m.type in ("human", "ai") and m.content]
    messages_text = "\n".join(f"{m.type.upper()}: {str(m.content)[:300]}" for m in recent)

    response = _llm_small.invoke([
        SystemMessage(content=PROFILE_UPDATE_PROMPT.format(
            current_profile=json.dumps(profile, indent=2),
            recent_messages=messages_text,
        )),
        HumanMessage(content="Update the profile based on the conversation above. Return only the JSON object."),
    ])

    match = re.search(r"\{.*\}", response.content, re.DOTALL)
    if match is None:
        print(f"[Profile] WARNING: LLM returned no JSON — raw response: {response.content[:300]}")
        return {}
    try:
        updated = json.loads(match.group())
        save_profile(session_id, updated)
        print(f"[Profile] Updated for session '{session_id}'")
    except json.JSONDecodeError as e:
        print(f"[Profile] WARNING: JSON parse failed ({e}) — raw match: {match.group()[:300]}")

    return {}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------


def route_after_router(state: AgentState) -> str:
    return state.get("query_type", "out_of_scope")


def route_after_agent(state: AgentState) -> str:
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return "end"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Build graph  (module-level `graph` required by langgraph.json)
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    builder = StateGraph(AgentState)

    builder.add_node("router",         router)
    builder.add_node("agent",          agent)
    builder.add_node("tools",          ToolNode(ALL_TOOLS))
    builder.add_node("decline",        decline)
    builder.add_node("update_profile", update_profile)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "structured":   "agent",
            "unstructured": "agent",
            "out_of_scope": "decline",
        },
    )
    builder.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "end": "update_profile"},
    )
    builder.add_edge("tools",          "agent")
    builder.add_edge("update_profile", END)
    builder.add_edge("decline",        END)

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()  # module-level instance for LangGraph Studio
