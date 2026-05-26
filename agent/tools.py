import json
import os
from functools import lru_cache
from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Bitext-customer-support-llm-chatbot-training-dataset",
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv",
)


@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    df = pd.read_csv(_CSV_PATH, encoding="utf-8")
    df["category"] = df["category"].str.upper()
    df["intent"] = df["intent"].str.lower()
    return df


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------


class FilterInput(BaseModel):
    category: Optional[str] = Field(
        None,
        description="Filter by category name (e.g. 'REFUND', 'ORDER'). Case-insensitive. Leave None to include all categories.",
    )
    intent: Optional[str] = Field(
        None,
        description="Filter by intent slug (e.g. 'get_refund', 'cancel_order'). Case-insensitive. Leave None to include all intents.",
    )


class SamplesInput(BaseModel):
    n: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of example rows to return (1–20).",
    )
    category: Optional[str] = Field(
        None,
        description="Filter by category. Case-insensitive.",
    )
    intent: Optional[str] = Field(
        None,
        description="Filter by intent slug. Case-insensitive.",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def list_categories() -> str:
    """Return all unique category names in the dataset (e.g. ACCOUNT, REFUND, ORDER).
    Use this when the user asks what categories or topics exist in the dataset."""
    df = _load_df()
    return json.dumps(sorted(df["category"].unique().tolist()))


@tool(args_schema=FilterInput)
def list_intents(category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Return all unique intent slugs, optionally filtered to one category.
    Use this to discover valid intent names before filtering — e.g. to answer
    'show examples of people wanting their money back', call this first to find
    the slug 'get_refund', then pass it to get_samples or count_records."""
    df = _load_df()
    if category and category.lower() != "null":
        df = df[df["category"] == category.upper()]
    return json.dumps(sorted(df["intent"].unique().tolist()))


@tool(args_schema=FilterInput)
def count_records(category: Optional[str] = None, intent: Optional[str] = None) -> int:
    """Count how many records match the given filters.
    Use for any 'how many' question — e.g. 'How many refund requests did we get?'
    Both filters are optional; omitting one means no filter on that column."""
    df = _load_df()
    if category and category.lower() != "null":
        df = df[df["category"] == category.upper()]
    if intent and intent.lower() != "null":
        df = df[df["intent"] == intent.lower()]
    return len(df)


@tool(args_schema=SamplesInput)
def get_samples(n: int = 5, category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Retrieve N example records (instruction + response pairs) from the dataset.
    Use for 'show me examples' or open-ended summarisation questions like
    'How do agents respond to cancellation requests?' — pass a larger n (up to 20)
    for summarisation so there is enough context to synthesise patterns."""
    df = _load_df()
    if category and category.lower() != "null":
        df = df[df["category"] == category.upper()]
    if intent and intent.lower() != "null":
        df = df[df["intent"] == intent.lower()]
    if df.empty:
        return json.dumps([])
    sample = df[["instruction", "category", "intent", "response"]].sample(min(n, len(df)))
    return json.dumps(sample.to_dict(orient="records"), ensure_ascii=False)


@tool(args_schema=FilterInput)
def get_intent_distribution(category: Optional[str] = None, intent: Optional[str] = None) -> str:
    """Return a count of records per intent, optionally scoped to one category.
    Use for 'distribution' or 'breakdown' questions — e.g.
    'What is the distribution of intents in the ACCOUNT category?'"""
    df = _load_df()
    if category and category.lower() != "null":
        df = df[df["category"] == category.upper()]
    return json.dumps(df["intent"].value_counts().to_dict())


ALL_TOOLS = [
    list_categories,
    list_intents,
    count_records,
    get_samples,
    get_intent_distribution,
]
