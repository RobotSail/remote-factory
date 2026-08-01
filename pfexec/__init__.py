"""pfexec — probabilistic workflow execution engine.

Treats workflow steps as inference over latent variables using particle-based
belief tracking, Thompson sampling, and Bradley-Terry scoring.
"""

from pfexec.engine import EngineConfig, EngineResult
from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec
from pfexec.state import ExecutionState

__all__ = [
    "EdgeSpec",
    "EngineConfig",
    "EngineResult",
    "ExecutionState",
    "NodeSpec",
    "WorkflowSpec",
]
