"""Factory SKILL.md baseline — single-prompt execution for comparison.

Converts a pfexec WorkflowSpec into a factory-style SKILL.md prose prompt
and runs the entire workflow in a single claude --bare call. This replicates
how the factory system executes workflows (one LLM session with a prose
playbook) as a comparison baseline for pfexec's programmatic execution.
"""

from __future__ import annotations

import re
import subprocess

from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import WorkflowSpec
from pfexec.state import Belief, ExecutionState, Particle, TraceNode, TraceTree


def _topo_order(workflow: WorkflowSpec) -> list[str]:
    adj: dict[str, list[str]] = {n.id: [] for n in workflow.nodes}
    in_degree: dict[str, int] = {n.id: 0 for n in workflow.nodes}
    for e in workflow.edges:
        adj[e.source].append(e.target)
        in_degree[e.target] = in_degree.get(e.target, 0) + 1

    queue = [workflow.entry] if workflow.entry else [
        nid for nid, deg in in_degree.items() if deg == 0
    ]
    order: list[str] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in adj.get(node, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return order


def _terminal_nodes(workflow: WorkflowSpec) -> list[str]:
    sources = {e.source for e in workflow.edges}
    return [n.id for n in workflow.nodes if n.id not in sources]


def generate_skill_md(workflow: WorkflowSpec) -> str:
    """Convert a WorkflowSpec into a factory-style SKILL.md prose prompt."""
    node_map = {n.id: n for n in workflow.nodes}
    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    lines: list[str] = [
        "---",
        f"name: {workflow.name}",
        f'description: "Execute the {workflow.name} workflow as a single-pass pipeline."',
        "---",
        "",
        f"# {workflow.name}",
        "",
        "You are executing a multi-step reasoning workflow. Follow each phase "
        "in order. For each phase, use the output of the previous phase as "
        "context (replacing {input} references).",
        "",
        "**Output format:** After completing each phase, write your result "
        "under a clearly marked header:",
        "```",
        "### Output: <node_id>",
        "<your result here>",
        "```",
        "",
        "After all phases are complete, provide a final consolidated answer "
        "under `### Final Answer`.",
        "",
    ]

    for i, nid in enumerate(order, 1):
        node = node_map[nid]
        lines.append(f"## Phase {i}: {nid}")
        lines.append("")
        lines.append(f"**Role:** {node.spec}")
        lines.append("")
        lines.append("**Task:**")
        lines.append(node.theta_prior)
        lines.append("")
        if i == 1:
            lines.append(
                "The `{input}` above will be provided in the user message."
            )
        else:
            prev_nid = order[i - 2]
            lines.append(
                f"Use the output from Phase {i - 1} (`{prev_nid}`) as "
                f"the `{{input}}` for this phase."
            )
        lines.append("")
        lines.append(f"Write your result under `### Output: {nid}`")
        lines.append("")

    lines.append("## Completion")
    lines.append("")
    lines.append(
        f"After completing all {len(order)} phases, read your output from "
        f"the final phase (`{terminal_id}`) and provide it under "
        f"`### Final Answer`."
    )
    lines.append("")

    return "\n".join(lines)


def parse_skill_output(raw_output: str, workflow: WorkflowSpec) -> dict[str, str]:
    """Extract per-node outputs from SKILL.md-style LLM response.

    Scans for '### Output: <node_id>' sections and returns {node_id: text}.
    """
    node_ids = {n.id for n in workflow.nodes}
    results: dict[str, str] = {}

    pattern = re.compile(r"###\s+Output:\s*(\S+)")
    matches = list(pattern.finditer(raw_output))

    for i, match in enumerate(matches):
        node_id = match.group(1)
        if node_id not in node_ids:
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_output)
        section = raw_output[start:end]
        final_marker = section.find("### Final Answer")
        if final_marker != -1:
            section = section[:final_marker]
        results[node_id] = section.strip()

    return results


def _extract_final_answer(raw_output: str) -> str:
    """Extract the ### Final Answer section from LLM output."""
    marker = "### Final Answer"
    idx = raw_output.rfind(marker)
    if idx == -1:
        return ""
    text = raw_output[idx + len(marker):]
    text = text.strip().lstrip(":").strip()
    return text.strip()


def run_factory_baseline(
    workflow: WorkflowSpec,
    user_input: str,
    config: EngineConfig,
) -> EngineResult:
    """Run a workflow as a single claude --bare call with SKILL.md system prompt."""
    skill_md = generate_skill_md(workflow)

    user_prompt = f"Execute the workflow for the following input:\n\n{user_input}"

    cmd = [
        "claude", "--bare",
        "--disallowedTools",
        "Bash Read Edit Write Agent NotebookEdit WebFetch WebSearch",
        "--system-prompt", skill_md,
        "-p", user_prompt,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude failed (exit {result.returncode}): {result.stderr}")

    raw_output = result.stdout.strip()
    node_outputs = parse_skill_output(raw_output, workflow)
    final_answer = _extract_final_answer(raw_output)

    order = _topo_order(workflow)
    if not final_answer and node_outputs:
        for nid in reversed(order):
            if nid in node_outputs:
                final_answer = node_outputs[nid]
                break
    if not final_answer:
        final_answer = raw_output.split("\n")[-1].strip()

    all_outputs = [node_outputs[nid] for nid in order if nid in node_outputs]

    belief = Belief(particles=[Particle(brief="baseline", weight=1.0)])
    trace = TraceTree(root=TraceNode(node_id="root"))
    state = ExecutionState(
        pointer=order[-1] if order else "",
        belief=belief,
        trace=trace,
        step=len(node_outputs),
        budget_remaining=config.max_steps - len(node_outputs),
        user_input=user_input,
        node_outputs=node_outputs,
    )

    return EngineResult(
        final_state=state,
        output=final_answer,
        steps_taken=len(node_outputs),
        forks_triggered=0,
        terminated_by="complete",
        all_outputs=all_outputs,
    )
