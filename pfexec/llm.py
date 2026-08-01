"""LLM interface — pluggable backends for prompt execution."""

from __future__ import annotations

import subprocess
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    def call(self, prompt: str, system: str = "") -> str: ...


class ClaudeBackend:
    def __init__(self, cli: str = "claude", timeout: int = 120):
        self._cli = cli
        self._timeout = timeout

    def call(self, prompt: str, system: str = "") -> str:
        cmd = [self._cli, "-p", prompt]
        if system:
            cmd.extend(["--system", system])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{self._cli} failed (exit {result.returncode}): {result.stderr}")
        return result.stdout.strip()


class DeterministicBackend:
    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default: str = "ok",
    ):
        self._responses = responses or {}
        self._default = default

    def call(self, prompt: str, system: str = "") -> str:
        for substring, response in self._responses.items():
            if substring in prompt:
                return response
        return self._default


def get_backend(mode: str = "claude", **kwargs) -> LLMBackend:
    if mode == "mock":
        return DeterministicBackend(**kwargs)
    return ClaudeBackend(**kwargs)
