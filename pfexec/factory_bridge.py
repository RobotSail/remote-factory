"""Bridge: compile factory Workflow -> pfexec WorkflowSpec.

Maps factory node types to pfexec NodeSpec:
- AgentNode -> NodeSpec (role as spec, prompt_template as theta_prior)
- GateNode -> NodeSpec (effectful when evaluator_command present)
- FnNode -> NodeSpec (command as theta_prior)
- Study -> NodeSpec (study command as theta_prior)
- ForkNode -> flattened (targets inlined sequentially)
- JoinNode -> skipped (barrier handled by sequential ordering)
"""

from __future__ import annotations

from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


def compile_workflow(factory_workflow) -> WorkflowSpec:
    """Compile a factory Workflow to pfexec WorkflowSpec.

    Args:
        factory_workflow: A factory.workflow.primitives.Workflow instance

    Returns:
        pfexec WorkflowSpec ready for execution
    """
    from factory.workflow.primitives import (
        ForkNode,
        JoinNode,
        SelectionNode,
        SubgraphForkNode,
        VerdictType,
    )
    from factory.workflow.skill_export import _topological_sort

    nodes: list[NodeSpec] = []
    edges: list[EdgeSpec] = []
    skip_ids: set[str] = set()

    for node in factory_workflow.nodes.values():
        if isinstance(node, ForkNode):
            skip_ids.update(node.targets)

    topo_order = _topological_sort(factory_workflow)

    for nid in topo_order:
        node = factory_workflow.nodes[nid]

        if isinstance(node, ForkNode):
            for target_id in node.targets:
                target = factory_workflow.nodes[target_id]
                nodes.append(_convert_node(target_id, target))
            continue

        if isinstance(node, JoinNode):
            continue

        if isinstance(node, (SubgraphForkNode, SelectionNode)):
            nodes.append(NodeSpec(
                id=nid,
                spec=f"Execute {nid} (parallel subgraph)",
                theta_prior=f"Plan and coordinate the {nid} subgraph. {{input}}",
            ))
            continue

        if nid in skip_ids:
            continue

        nodes.append(_convert_node(nid, node))

    edge_node_ids = {n.id for n in nodes}

    for edge in factory_workflow.edges:
        if edge.condition == VerdictType.RELOOP:
            continue
        if edge.source in edge_node_ids and edge.target in edge_node_ids:
            edges.append(EdgeSpec(source=edge.source, target=edge.target))

    for nid in topo_order:
        node = factory_workflow.nodes[nid]
        if not isinstance(node, ForkNode):
            continue

        if len(node.targets) > 1:
            for i in range(len(node.targets) - 1):
                src = node.targets[i]
                tgt = node.targets[i + 1]
                if src in edge_node_ids and tgt in edge_node_ids:
                    edges.append(EdgeSpec(source=src, target=tgt))

        # Connect incoming edges to first fork target
        if node.targets:
            first_target = node.targets[0]
            for e in factory_workflow.edges:
                if e.target == nid and e.source in edge_node_ids and first_target in edge_node_ids:
                    edges.append(EdgeSpec(source=e.source, target=first_target))

        # Connect last fork target to whatever follows the join
        if node.targets:
            last_target = node.targets[-1]
            for e in factory_workflow.edges:
                join_node = factory_workflow.nodes.get(e.target)
                if isinstance(join_node, JoinNode) and set(join_node.sources) & set(node.targets):
                    for e2 in factory_workflow.edges:
                        if e2.source == join_node.id and e2.target in edge_node_ids:
                            edges.append(EdgeSpec(source=last_target, target=e2.target))

    seen: set[tuple[str, str]] = set()
    unique_edges: list[EdgeSpec] = []
    for e in edges:
        key = (e.source, e.target)
        if key not in seen:
            seen.add(key)
            unique_edges.append(e)

    entry = nodes[0].id if nodes else factory_workflow.start_node

    return WorkflowSpec(
        name=factory_workflow.name,
        nodes=nodes,
        edges=unique_edges,
        entry=entry,
    )


def _convert_node(nid: str, node) -> NodeSpec:
    """Convert a factory node to pfexec NodeSpec."""
    from factory.workflow.primitives import AgentNode, FnNode, GateNode, Study

    if isinstance(node, Study):
        cmd = node.command.replace("{project_path}", "{project_path}")
        return NodeSpec(
            id=nid,
            spec="Run local study to gather observations",
            theta_prior=f"Run: {cmd}\nReport observations. {{input}}",
        )

    if isinstance(node, AgentNode):
        role = node.role.value
        spec = f"{role}: {node.prompt_template[:100]}" if node.prompt_template else f"{role} agent"
        theta_prior = node.prompt_template or f"Execute the {role} task. {{input}}"
        return NodeSpec(
            id=nid,
            spec=spec,
            theta_prior=theta_prior,
        )

    if isinstance(node, GateNode):
        spec = f"Gate: {node.gate_prompt[:100]}" if node.gate_prompt else f"Gate {nid}"
        theta_prior = node.gate_prompt or f"Evaluate gate {nid}. {{input}}"
        if node.evaluator_command:
            theta_prior = f"Run: {node.evaluator_command}\n\nThen: {theta_prior}"
        return NodeSpec(
            id=nid,
            spec=spec,
            theta_prior=theta_prior,
            effect="effectful" if node.evaluator_command else "pure",
        )

    if isinstance(node, FnNode):
        cmd = node.command.replace("{project_path}", "{project_path}")
        spec = node.notes[:100] if node.notes else f"Run {nid}"
        return NodeSpec(
            id=nid,
            spec=spec,
            theta_prior=f"Run: {cmd}\n{{input}}",
        )

    return NodeSpec(
        id=nid,
        spec=f"Execute {nid}",
        theta_prior=f"Execute the {nid} step. {{input}}",
    )


def list_workflows() -> list[str]:
    """List all available factory workflow names."""
    from factory.workflow.definitions import register_all

    return list(register_all().keys())


def get_workflow(name: str):
    """Get a factory workflow by name."""
    from factory.workflow.definitions import register_all

    workflows = register_all()
    if name not in workflows:
        raise ValueError(f"Unknown workflow: {name}. Available: {list(workflows.keys())}")
    return workflows[name]
