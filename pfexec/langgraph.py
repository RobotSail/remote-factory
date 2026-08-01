"""LangGraph compiler — converts pfexec IR to LangGraph StateGraph."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from pfexec.engine import EngineConfig, EngineResult, _suffix_score
from pfexec.ir import WorkflowSpec
from pfexec.llm import LLMBackend
from pfexec.primitives import fork, init, observe, sample
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


class PfExecState(TypedDict):
    belief: dict
    trace: dict
    pointer: str
    step: int
    outputs: list[str]
    fork_count: int
    budget: int


def _belief_to_dict(belief: Belief) -> dict:
    return {
        "particles": [
            {"brief": p.brief, "weight": p.weight, "evidence": p.evidence}
            for p in belief.particles
        ]
    }


def _dict_to_belief(d: dict) -> Belief:
    return Belief(
        particles=[
            Particle(brief=p["brief"], weight=p["weight"], evidence=p.get("evidence", ""))
            for p in d.get("particles", [])
        ]
    )


def _trace_node_to_dict(node: TraceNode) -> dict:
    return {
        "node_id": node.node_id,
        "checkpoint_id": node.checkpoint_id,
        "alive": node.alive,
        "summary": node.summary,
        "children": [_trace_node_to_dict(c) for c in node.children],
    }


def _dict_to_trace_node(d: dict) -> TraceNode:
    return TraceNode(
        node_id=d["node_id"],
        checkpoint_id=d.get("checkpoint_id", ""),
        alive=d.get("alive", True),
        summary=d.get("summary", ""),
        children=[_dict_to_trace_node(c) for c in d.get("children", [])],
    )


def _state_to_pfexec(s: PfExecState, budget: int = 50) -> ExecutionState:
    belief = _dict_to_belief(s["belief"])
    root = _dict_to_trace_node(s["trace"])
    return ExecutionState(
        pointer=s["pointer"],
        belief=belief,
        trace=TraceTree(root=root),
        step=s["step"],
        budget_remaining=s.get("budget", budget),
    )


def _pfexec_to_state(es: ExecutionState, outputs: list[str], fork_count: int) -> PfExecState:
    return PfExecState(
        belief=_belief_to_dict(es.belief),
        trace=_trace_node_to_dict(es.trace.root),
        pointer=es.pointer,
        step=es.step,
        outputs=outputs,
        fork_count=fork_count,
        budget=es.budget_remaining,
    )


def compile(
    workflow: WorkflowSpec,
    backend: LLMBackend,
    config: EngineConfig | None = None,
) -> StateGraph:
    cfg = config or EngineConfig()
    node_map = {n.id: n for n in workflow.nodes}
    successors = {}
    for n in workflow.nodes:
        successors[n.id] = [e.target for e in workflow.edges if e.source == n.id]

    def _make_node_fn(nid: str):
        def node_fn(state: PfExecState) -> dict:
            es = _state_to_pfexec(state, cfg.max_steps)
            node = node_map[nid]
            es, output = sample(es, node, backend)
            es = observe(es, output, backend)
            es.pointer = nid
            outputs = list(state["outputs"]) + [output]
            fc = state["fork_count"]

            score = _suffix_score(es.belief)
            if score < cfg.tau and fc < cfg.max_forks:
                es = fork(es, cfg.rewind_steps, backend)
                fc += 1

            result = _pfexec_to_state(es, outputs, fc)
            return dict(result)
        return node_fn

    def _make_router(nid: str):
        succs = successors[nid]
        def router(state: PfExecState) -> str:
            if state["budget"] <= 0:
                return END
            if state["fork_count"] >= cfg.max_forks:
                return END
            pointer = state["pointer"]
            if pointer != nid and pointer in node_map:
                return pointer
            if succs:
                return succs[0]
            return END
        return router

    graph = StateGraph(PfExecState)

    for nid in node_map:
        graph.add_node(nid, _make_node_fn(nid))

    graph.add_edge(START, workflow.entry)

    for nid in node_map:
        succs = successors[nid]
        if not succs:
            graph.add_edge(nid, END)
        elif len(succs) == 1:
            has_fork_possible = True
            graph.add_conditional_edges(nid, _make_router(nid))
        else:
            graph.add_conditional_edges(nid, _make_router(nid))

    return graph


def run_compiled(
    graph: StateGraph,
    workflow: WorkflowSpec,
    user_input: str,
    backend: LLMBackend,
    config: EngineConfig | None = None,
) -> EngineResult:
    cfg = config or EngineConfig()
    es = init(workflow, user_input, cfg.n_particles, backend)
    es.budget_remaining = cfg.max_steps

    initial_state = _pfexec_to_state(es, [], 0)

    checkpointer = MemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    thread_id = str(uuid.uuid4())
    result = app.invoke(
        dict(initial_state),
        config={"configurable": {"thread_id": thread_id}},
    )

    final_es = _state_to_pfexec(result, cfg.max_steps)
    outputs = result.get("outputs", [])
    forks = result.get("fork_count", 0)

    if final_es.budget_remaining <= 0:
        terminated_by = "budget"
    elif forks >= cfg.max_forks:
        terminated_by = "max_forks"
    else:
        terminated_by = "complete"

    return EngineResult(
        final_state=final_es,
        output="\n".join(outputs),
        steps_taken=cfg.max_steps - final_es.budget_remaining,
        forks_triggered=forks,
        terminated_by=terminated_by,
    )
