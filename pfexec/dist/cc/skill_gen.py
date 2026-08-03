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
                     session_dir: Path) -> str:
    node_map = {n.id: n for n in workflow.nodes}
    order = _topo_order(workflow)
    terminal = _terminal_nodes(workflow)
    terminal_id = terminal[0] if terminal else order[-1]

    lines: list[str] = []
    lines.append(f"# {workflow.name} — pfexec Agentic Execution")
    lines.append("")
    lines.append("You are executing a pfexec workflow. You have access to pfexec CLI tools via Bash.")
    lines.append("")

    lines.append("## Available Tools")
    lines.append("")
    lines.append("All tools are invoked via `python -m pfexec.dist.cc.belief_io`:")
    lines.append("")
    lines.append("### Initialize")
    lines.append("```bash")
    lines.append(
        f"python -m pfexec.dist.cc.belief_io init "
        f"--session {session_dir} --workflow {session_dir}/workflow.json "
        f"--input \"...\" --particles {config.n_particles} --backend claude"
    )
    lines.append("```")
    lines.append("")
    lines.append("### Before each node — Sample")
    lines.append("```bash")
    lines.append(
        f"python -m pfexec.dist.cc.belief_io sample "
        f"--session {session_dir} --node <node_id> --backend claude"
    )
    lines.append("```")
    lines.append("Reads belief state, writes hooks/prompt.txt with conditioned prompt.")
    lines.append("")
    lines.append("### After each node — Observe")
    lines.append("```bash")
    lines.append(
        f"python -m pfexec.dist.cc.belief_io observe "
        f"--session {session_dir} --node <node_id> --backend claude"
    )
    lines.append("```")
    lines.append("First write your output to node_outputs/<node_id>.txt, then run observe.")
    lines.append("")
    lines.append("### After effectful nodes — Fork Check")
    lines.append("```bash")
    lines.append(
        f"python -m pfexec.dist.cc.belief_io fork-check "
        f"--session {session_dir} --node <node_id> "
        f"--tau {config.tau} --max-forks {config.max_forks} --backend claude"
    )
    lines.append("```")
    lines.append("Prints FORK or CONTINUE.")
    lines.append("")

    lines.append("## Workflow Nodes (execute in order)")
    lines.append("")
    for nid in order:
        node = node_map[nid]
        lines.append(f"- **{nid}**: role=`{node.spec}`, effect=`{node.effect}`")
    lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.append("For each node in order:")
    lines.append("")
    for i, nid in enumerate(order, 1):
        node = node_map[nid]
        lines.append(f"### Step {i}: {nid}")
        lines.append("")
        lines.append(f"1. Run: `python -m pfexec.dist.cc.belief_io sample "
                     f"--session {session_dir} --node {nid} --backend claude`")
        lines.append("2. Read `hooks/prompt.txt` for the conditioned prompt.")
        lines.append("3. Execute the task described in the prompt.")
        lines.append(f"4. Write your output to `node_outputs/{nid}.txt` "
                     "using the Write tool or echo.")
        lines.append(f"5. Run: `python -m pfexec.dist.cc.belief_io observe "
                     f"--session {session_dir} --node {nid} --backend claude`")
        if node.effect == "effectful":
            lines.append(f"6. Run: `python -m pfexec.dist.cc.belief_io fork-check "
                         f"--session {session_dir} --node {nid} "
                         f"--tau {config.tau} --max-forks {config.max_forks} --backend claude`")
            lines.append("   - If FORK: read `state.json` for the rewound pointer "
                         "and go back to that node.")
            lines.append("   - If CONTINUE: proceed to the next step.")
        else:
            lines.append("6. Continue to the next step.")
        lines.append("")

    lines.append("## Output")
    lines.append("")
    lines.append(f"After all nodes complete, report the content of "
                 f"`node_outputs/{terminal_id}.txt`.")
    lines.append("")

    return "\n".join(lines)
