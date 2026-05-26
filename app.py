"""Streamlit chat UI for the Customer Service Data Analyst Agent (Bonus A).

Run with:
    streamlit run app.py
"""

import os
import sqlite3

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import build_graph

_DB_PATH = os.path.join(os.path.dirname(__file__), "agent", "storage", "checkpoints.db")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="CS Agent", page_icon="📊", layout="wide")
st.title("📊 Customer Service Data Analyst Agent")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session")
    session_id = st.text_input(
        "Session ID",
        value="default",
        help="Reuse the same ID to restore conversation history across restarts.",
    )
    st.caption("Each session has its own conversation history and user profile.")
    st.divider()
    st.markdown("**Example queries**")
    st.markdown("- What categories exist?")
    st.markdown("- How many refund requests did we get?")
    st.markdown("- Show me 5 examples from SHIPPING")
    st.markdown("- Summarize the FEEDBACK category")
    st.markdown("- What is the distribution of intents in ACCOUNT?")
    st.markdown("- What do you remember about me?")

# ---------------------------------------------------------------------------
# Initialise graph once per Streamlit session
# ---------------------------------------------------------------------------

if "graph" not in st.session_state:
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    st.session_state.graph = build_graph(checkpointer=checkpointer)

# Clear display messages when the user switches session ID
if st.session_state.get("current_session_id") != session_id:
    st.session_state.current_session_id = session_id
    st.session_state.display_messages = []

# ---------------------------------------------------------------------------
# Render existing chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.get("display_messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("reasoning"):
            with st.expander("Reasoning steps"):
                for step in msg["reasoning"]:
                    st.text(step)

# ---------------------------------------------------------------------------
# Handle new input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about the customer service dataset..."):
    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.display_messages.append({"role": "user", "content": prompt})

    config = {"configurable": {"thread_id": session_id}}
    reasoning_steps = []
    final_answer = ""
    prev_len = 0

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for state in st.session_state.graph.stream(
                {"messages": [HumanMessage(content=prompt)]},
                config=config,
                stream_mode="values",
            ):
                messages = state.get("messages", [])
                # Only process messages added in this event
                new_msgs = messages[prev_len:]
                prev_len = len(messages)

                for m in new_msgs:
                    if m.type == "ai" and hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            reasoning_steps.append(f"🔧 Tool call: {tc['name']}  {tc['args']}")
                    elif m.type == "tool":
                        reasoning_steps.append(f"📊 {m.name}: {str(m.content)[:300]}")
                    elif m.type == "ai" and m.content:
                        final_answer = m.content

        st.markdown(final_answer)
        if reasoning_steps:
            with st.expander("Reasoning steps"):
                for step in reasoning_steps:
                    st.text(step)

    st.session_state.display_messages.append({
        "role": "assistant",
        "content": final_answer,
        "reasoning": reasoning_steps,
    })
