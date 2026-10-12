"""The research session as a LangGraph StateGraph: wiring only.

Steps, state and topology come from `nodes.py`. This module turns them into nodes,
static edges and conditional edges, and maps "ask the user" onto LangGraph's `interrupt`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy, interrupt

from forage.experimental.langgraph_engine.nodes import (
    ALWAYS_NEXT, BRANCHES, ENTRY, NODE_NAMES, STEPS, EngineContext, SessionState, run_step,
)

__all__ = ["NODE_NAMES", "build_graph"]


def _node(name: str) -> Callable[[SessionState, Runtime[EngineContext]], dict[str, Any]]:
    def node(state: SessionState, runtime: Runtime[EngineContext]) -> dict[str, Any]:
        return run_step(name, state, replace(runtime.context, ask=interrupt))

    node.__name__ = name
    return node


def _route(state: SessionState) -> str:
    return state["route"]


def build_graph(retry: dict[str, RetryPolicy] | None = None) -> StateGraph:
    """The uncompiled graph. `retry` optionally attaches a LangGraph retry policy to named nodes."""
    graph = StateGraph(SessionState, context_schema=EngineContext)
    for name in STEPS:
        graph.add_node(name, _node(name), retry_policy=(retry or {}).get(name))
    graph.add_edge(START, ENTRY)
    for name, target in ALWAYS_NEXT.items():
        graph.add_edge(name, END if target is None else target)
    for name, choices in BRANCHES.items():
        graph.add_conditional_edges(name, _route, {choice: choice for choice in choices})
    return graph
