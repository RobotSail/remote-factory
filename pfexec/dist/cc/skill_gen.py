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

    lines: list[str] = []
    lines.append(f"# {workflow.name} — pfexec Agentic Protocol")
    lines.append("")
    lines.append(
        "You are executing a pfexec probabilistic workflow. "
        "Follow this protocol EXACTLY for each node."
    )
    lines.append("")

    lines.append("## Session Directory")
    lines.append(f"All paths are relative to: {session_dir}")
    lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.append("For EACH node listed below, in order:")
    lines.append("")

    lines.append("### Before the node")
    lines.append("Run this command to get the conditioned prompt:")
    lines.append("```bash")
    lines.append(
        f"python3 -m pfexec.dist.cc.belief_io sample "
        f"--session {session_dir} --node <NODE_ID> --backend {backend_mode}"
    )
    lines.append(f"cat {session_dir}/hooks/prompt.txt")
    lines.append("```")
    lines.append("Read the output of prompt.txt — this is your task instruction for this node.")
    lines.append("")

    lines.append("### Execute the node")
    lines.append(
        "Perform the task described in prompt.txt. Think carefully and produce your answer."
    )
    lines.append("")

    lines.append("### After the node")
    lines.append("Write your output to the node output file:")
    lines.append("```bash")
    lines.append(f"cat > {session_dir}/node_outputs/<NODE_ID>.txt << PFEXEC_OUTPUT")
    lines.append("<YOUR OUTPUT HERE>")
    lines.append("PFEXEC_OUTPUT")
    lines.append("```")
    lines.append("")
    lines.append(
        "Note: A PostToolUse hook automatically runs observe and fork-check after you write."
    )
    lines.append("")
    lines.append("Then check the fork status:")
    lines.append("```bash")
    lines.append(f"cat {session_dir}/hooks/fork_status.txt")
    lines.append("```")
    lines.append(
        "- If it says FORK: read state.json to find the rewound pointer, "
        "then go back to that node and re-execute from there."
    )
    lines.append("- If it says CONTINUE: proceed to the next node.")
    lines.append("")

    lines.append("## Nodes (execute in this order)")
    lines.append("")
    for nid in order:
        node = node_map[nid]
        lines.append(f"### Node: {nid}")
        lines.append(f"- Role: {node.spec}")
        lines.append(f"- Effect: {node.effect}")
        lines.append("")

    lines.append("## Completion")
    lines.append("After all nodes are done, read the terminal node output:")
    lines.append("```bash")
    lines.append(f"cat {session_dir}/node_outputs/{terminal_id}.txt")
    lines.append("```")
    lines.append("Report this as your final answer.")
    lines.append("")

    return "\n".join(lines)
