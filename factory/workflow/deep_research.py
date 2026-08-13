"""Deep-research iterative research workflow with decomposition.

Runs study → decomposer → deep_researcher → CEO coverage gate.
The decomposer generates research directions; the researcher executes them.
Terminal mode — does not chain to build or improve.
Triggered via `factory workflow run deep-research` or
`factory ceo /path --mode deep-research`.
"""

from typing import Any

from factory.models import ProjectState
from factory.workflow.primitives import (
    AgentNode,
    AgentRole,
    ArtifactCheck,
    Edge,
    GateNode,
    Study,
    VerdictType,
    Workflow,
)

meta = {
    "name": "deep-research",
    "description": (
        "Iterative research with decomposition, faithfulness checking, and "
        "coverage evaluation. A decomposer generates research directions; "
        "the researcher executes them with multiple rounds of "
        "WebSearch/WebFetch, following an inside-out protocol."
    ),
}

_DECOMPOSER_PROMPT = (
    "You are the Research Decomposer. Produce 3-5 research directions tailored "
    "to the current mode and project context.\n\n"
    "Read:\n"
    "- The CEO's task (contains the original prompt and mode context)\n"
    "- .factory/strategy/observations.md (if exists — project state)\n"
    "- .factory/config.json (if exists — project config, research_target)\n\n"
    "Based on what you find, determine the research context:\n"
    "- New project (no .factory/) → web-focused directions (similar, tech, pitfalls)\n"
    "- Existing project, improve → mixed directions (internal assessment first, then "
    "targeted external search for weak dimensions)\n"
    "- Factory itself, create mode → code-focused directions (read existing patterns, "
    "parse mode intent, minimal web for novel patterns only)\n"
    "- Research target configured → failure-focused directions (within mutable surfaces)\n\n"
    "For each direction, write:\n\n"
    "### Direction N: [title]\n"
    "- **What to research:** specific question, not generic\n"
    "- **Why it matters:** how this connects to the original prompt and project\n"
    "- **Type:** internal (code/project reading), external (web search), or mixed\n"
    "- **Coverage signal:** how the researcher knows this direction is adequately covered\n\n"
    "Rules:\n"
    "- Directions must be derived from the ORIGINAL PROMPT\n"
    "- If the project already uses pytest, don't direct 'research testing frameworks'\n"
    "- Each direction should produce findings the strategist can act on\n"
    "- 3-5 directions maximum\n"
    "- Specify type (internal/external/mixed) so the researcher knows whether to "
    "read code or search the web\n\n"
    "Write to .factory/strategy/research-directions.md"
)

_DEEP_RESEARCHER_PROMPT = (
    "Mode 5: Deep Research. Follow the Deep Research protocol in your "
    "system prompt. Read research directions from "
    ".factory/strategy/research-directions.md."
)

_GATE_COVERAGE_PROMPT = (
    "Check the deep research report against the research directions.\n\n"
    "Read .factory/strategy/research-directions.md (what was asked for) and "
    ".factory/strategy/research-combined.md (what was produced).\n\n"
    "For each direction the decomposer specified:\n"
    "1. Is it covered in the research report?\n"
    "2. Is the coverage adequate (actually researched, not just mentioned)?\n"
    "3. Did the researcher stay within the direction's scope?\n\n"
    "Also check:\n"
    "4. Does the report trace back to the original prompt?\n"
    "5. Are findings grounded (connected to codebase, not generic advice)?\n"
    "6. Are claims cited with URLs or file paths?\n\n"
    "PROCEED if all directions are covered.\n"
    "RELOOP listing which directions are missing or inadequately covered."
)


def workflow() -> Workflow:
    """W₁₅: Deep Research Mode — decompose-then-research with coverage checking.

    Study → decomposer (generates research directions) →
    deep_researcher (executes directions with internal iteration) →
    gate_coverage (CEO safety net checking per-direction coverage).

    The decomposer produces 3-5 research directions. The researcher executes
    them using WebSearch/WebFetch with built-in faithfulness checking. The gate
    checks coverage against the original directions.

    Terminal mode — does not chain to build or improve.
    """
    nodes: dict[str, Any] = {}

    nodes["study"] = Study(
        id="study",
        command="factory study {project_path}",
        writes={".factory/strategy/observations.md"},
    )

    nodes["decomposer"] = AgentNode(
        id="decomposer",
        role=AgentRole.RESEARCHER,
        prompt_template=_DECOMPOSER_PROMPT,
        reads={".factory/strategy/observations.md"},
        writes={".factory/strategy/research-directions.md"},
        post_checks=[
            ArtifactCheck(
                path=".factory/strategy/research-directions.md",
                must_exist=True,
                min_size=200,
            )
        ],
        model="sonnet",
        timeout=120,
    )

    nodes["deep_researcher"] = AgentNode(
        id="deep_researcher",
        role=AgentRole.RESEARCHER,
        prompt_template=_DEEP_RESEARCHER_PROMPT,
        reads={
            ".factory/strategy/observations.md",
            ".factory/strategy/research-directions.md",
        },
        writes={".factory/strategy/research-combined.md"},
        post_checks=[
            ArtifactCheck(
                path=".factory/strategy/research-combined.md",
                must_exist=True,
                min_size=500,
            )
        ],
        timeout=1800,
    )

    nodes["gate_coverage"] = GateNode(
        id="gate_coverage",
        evaluator_type="agent",
        evaluator_role=AgentRole.CEO,
        gate_prompt=_GATE_COVERAGE_PROMPT,
        reads={
            ".factory/strategy/research-directions.md",
            ".factory/strategy/research-combined.md",
        },
    )

    edges = [
        Edge(source="study", target="decomposer"),
        Edge(source="decomposer", target="deep_researcher"),
        Edge(source="deep_researcher", target="gate_coverage"),
        Edge(source="gate_coverage", target="deep_researcher", condition=VerdictType.RELOOP),
    ]

    def trigger(state: ProjectState, ctx: dict[str, Any]) -> bool:
        return state == ProjectState.HAS_FACTORY and ctx.get("mode") == "deep-research"

    return Workflow(
        name="deep-research",
        nodes=nodes,
        edges=edges,
        start_node="study",
        trigger=trigger,
        terminal=True,
    )
