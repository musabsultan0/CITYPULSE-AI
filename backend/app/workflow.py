from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END
from backend.app.agents import (
    supervisor_agent,
    traffic_agent,
    flood_agent,
    environment_agent,
    infrastructure_agent,
    risk_agent,
    knowledge_agent,
    response_planner_agent,
    communication_agent,
)


class CityState(TypedDict, total=False):
    scenario: str
    zone: str
    readings: dict[str, Any]
    traffic_features: dict[str, Any] | None
    flood_features: dict[str, Any] | None
    traffic_context: dict[str, Any] | None
    weather_context: dict[str, Any] | None
    use_llm: bool | None
    supervisor_finding: dict
    traffic_finding: dict
    flood_finding: dict
    environment_finding: dict
    infrastructure_finding: dict
    risk_assessment: dict
    risk_finding: dict
    knowledge_finding: dict
    response_plan: dict
    response_finding: dict
    communication_finding: dict
    workflow_trace: list[str]


def build_graph():
    g = StateGraph(CityState)

    def wrap(fn, key, finding_key):
        def node(s):
            out = fn(s)
            mode = out.get(finding_key, {}).get("reasoning_mode", "completed")
            return {
                **out,
                "workflow_trace": s.get("workflow_trace", []) + [f"{key}: completed [{mode}]"],
            }
        return node

    nodes = [
        ("supervisor", supervisor_agent, "supervisor_finding"),
        ("traffic", traffic_agent, "traffic_finding"),
        ("flood", flood_agent, "flood_finding"),
        ("environment", environment_agent, "environment_finding"),
        ("infrastructure", infrastructure_agent, "infrastructure_finding"),
        ("risk", risk_agent, "risk_finding"),
        ("knowledge", knowledge_agent, "knowledge_finding"),
        ("planner", response_planner_agent, "response_finding"),
        ("communication", communication_agent, "communication_finding"),
    ]

    for name, fn, fkey in nodes:
        g.add_node(name, wrap(fn, name, fkey))

    g.add_edge(START, "supervisor")
    order = [n[0] for n in nodes]
    for a, b in zip(order[:-1], order[1:]):
        g.add_edge(a, b)
    g.add_edge(order[-1], END)
    return g.compile()


CITY_GRAPH = build_graph()


def run_workflow(payload: dict) -> dict:
    return CITY_GRAPH.invoke({**payload, "workflow_trace": []})
