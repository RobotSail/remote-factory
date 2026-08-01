"""Shared fixtures for pfexec tests."""

from __future__ import annotations

import pytest

from pfexec.ir import EdgeSpec, NodeSpec, WorkflowSpec


@pytest.fixture
def linear_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="linear",
        nodes=[
            NodeSpec(id="a", spec="step A", theta_prior="Do A: {input}"),
            NodeSpec(id="b", spec="step B", theta_prior="Do B: {input}"),
            NodeSpec(id="c", spec="step C", theta_prior="Do C: {input}"),
        ],
        edges=[
            EdgeSpec(source="a", target="b"),
            EdgeSpec(source="b", target="c"),
        ],
        entry="a",
    )


@pytest.fixture
def branching_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        name="branching",
        nodes=[
            NodeSpec(id="start", spec="start", theta_prior="Begin: {input}"),
            NodeSpec(id="left", spec="left branch", theta_prior="Left: {input}"),
            NodeSpec(id="right", spec="right branch", theta_prior="Right: {input}"),
            NodeSpec(id="end", spec="end", theta_prior="End: {input}"),
        ],
        edges=[
            EdgeSpec(source="start", target="left", condition="go_left"),
            EdgeSpec(source="start", target="right", condition="go_right"),
            EdgeSpec(source="left", target="end"),
            EdgeSpec(source="right", target="end"),
        ],
        entry="start",
    )
