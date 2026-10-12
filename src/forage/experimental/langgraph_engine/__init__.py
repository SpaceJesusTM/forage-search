"""EXPERIMENTAL: one research session run step by step, with two interchangeable drivers.

Built to evaluate LangGraph as the session orchestrator (see
docs/LANGGRAPH_ARCHITECTURE_DECISION.md).

- `steps.py`   a session that can be rebuilt from a small dict (no framework)
- `nodes.py`   the steps and the control flow between them (no framework)
- `graph.py`, `engine.py`   LangGraph driver: StateGraph, SQLite checkpointer, interrupts
- `plain_runner.py`         comparison driver: the same steps from a plain loop

The LangGraph driver needs the optional `langgraph` and `langgraph-checkpoint-sqlite` packages.
"""

from forage.experimental.langgraph_engine.nodes import NODE_NAMES
from forage.experimental.langgraph_engine.steps import StepSession

__all__ = ["LangGraphEngine", "NODE_NAMES", "PlainStepEngine", "StepSession", "build_graph", "use_engine",
           "use_langgraph_engine"]


def __getattr__(name: str):
    # Imported on demand so the framework-free parts work without LangGraph installed.
    if name in ("LangGraphEngine", "use_langgraph_engine"):
        from forage.experimental.langgraph_engine import engine

        return getattr(engine, name)
    if name == "build_graph":
        from forage.experimental.langgraph_engine.graph import build_graph

        return build_graph
    if name in ("PlainStepEngine", "use_engine"):
        from forage.experimental.langgraph_engine import plain_runner

        return getattr(plain_runner, name)
    raise AttributeError(name)
