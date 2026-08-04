"""Generate SKILL.md playbook from pfexec IR."""

from __future__ import annotations

from pathlib import Path

from pfexec.engine import EngineConfig
from pfexec.ir import WorkflowSpec


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


def generate(workflow: WorkflowSpec, config: EngineConfig) -> str:
    node_map = {n.id: n for n in workflow.nodes}
    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    lines: list[str] = []
    lines.append(f"# {workflow.name}")
    lines.append("")
    lines.append("You are executing a pfexec workflow. Follow these steps exactly.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append('SESSION_DIR is the directory containing this SKILL.md file.')
    lines.append("")
    lines.append("## Workflow Nodes")
    lines.append("")

    for nid in order:
        node = node_map[nid]
        lines.append(f"### {nid}")
        lines.append(f"- **Role:** {node.spec}")
        lines.append(f"- **Effect:** {node.effect}")
        lines.append("")

    lines.append("## Execution")
    lines.append("")
    lines.append("For each node in order, do the following:")
    lines.append("")

    for i, nid in enumerate(order, 1):
        node = node_map[nid]
        lines.append(f"### Step {i}: {nid}")
        lines.append("")
        lines.append("1. Run the pre-step hook:")
        lines.append("   ```bash")
        lines.append(f"   bash hooks/pre_step.sh {nid}")
        lines.append("   ```")
        lines.append("2. Read `hooks/prompt.txt` for the conditioned prompt.")
        lines.append(f"3. Execute the task: **{node.spec}**")
        lines.append("   Use the prompt from `hooks/prompt.txt` as your instructions.")
        lines.append(f"4. Write your output to `node_outputs/{nid}.txt`")
        lines.append("5. Run the post-step hook:")
        lines.append("   ```bash")
        lines.append(f"   bash hooks/post_step.sh {nid}")
        lines.append("   ```")
        lines.append("6. Read `hooks/fork_status.txt`.")
        lines.append("   - If it says `FORK`, re-read `state.json` to find the rewound pointer,")
        lines.append("     then go back to the step for that node.")
        lines.append("   - If it says `CONTINUE`, proceed to the next step.")
        lines.append("")

    lines.append("## Output")
    lines.append("")
    lines.append(f"After all steps complete, read `node_outputs/{terminal_id}.txt`")
    lines.append("and report the final result to the user.")
    lines.append("")

    return "\n".join(lines)


def generate_agentic(workflow: WorkflowSpec, config: EngineConfig,
                     session_dir: Path, backend_mode: str = "claude") -> str:
    node_map = {n.id: n for n in workflow.nodes}
    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    lines: list[str] = [
        "---",
        f"name: {workflow.name}",
        f'description: "Execute the {workflow.name} workflow as a multi-phase pipeline."',
        "---",
        "",
        f"# {workflow.name} — pfexec Workflow",
        "",
        "You are executing a multi-step reasoning workflow. Follow each phase "
        "in order. For each phase, use the output of the previous phase as "
        "context (replacing {input} references).",
        "",
        "**Output format:** After completing each phase:",
        "1. Write your result under a `### Output: <node_id>` header in your response",
        f"2. Save it to the session directory:",
        "   ```",
        f"   Write to: {session_dir}/node_outputs/<node_id>.txt",
        "   ```",
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
        lines.append(
            f"Write your result under `### Output: {nid}` and save to "
            f"`node_outputs/{nid}.txt`"
        )
        lines.append("")

    lines.append("## Completion")
    lines.append("")
    lines.append(
        f"After completing all {len(order)} phases, provide your final "
        f"consolidated answer under `### Final Answer`."
    )
    lines.append("")

    lines.append("## Available Tools (optional)")
    lines.append("")
    lines.append(
        "You may use these to check belief state or trigger replanning:"
    )
    lines.append("")
    lines.append("- **pfexec sample**: Get a strategy hint conditioned on evidence so far")
    lines.append(f"  `bash hooks/pre_step.sh <node_id>` then read `hooks/hint.txt`")
    lines.append(
        "- **pfexec observe**: Manually update belief "
        "(runs automatically when you save outputs)"
    )
    lines.append(
        "- **pfexec fork-check**: Check if replanning is needed "
        "(runs automatically when you save outputs)"
    )
    lines.append("")
    lines.append(
        "These tools run automatically via hooks when you write to "
        "node_outputs/ — you do not need to call them manually unless "
        "you want explicit control."
    )
    lines.append("")

    return "\n".join(lines)
