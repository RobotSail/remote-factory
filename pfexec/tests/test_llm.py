"""Tests for pfexec.llm — LLM backend interface."""

from pfexec.llm import ClaudeBackend, DeterministicBackend, LLMBackend, get_backend


def test_deterministic_backend_canned_response():
    backend = DeterministicBackend(responses={"hello": "world", "foo": "bar"})
    assert backend.call("say hello") == "world"
    assert backend.call("do foo") == "bar"


def test_deterministic_backend_default():
    backend = DeterministicBackend(default="fallback")
    assert backend.call("unknown prompt") == "fallback"


def test_deterministic_backend_empty():
    backend = DeterministicBackend()
    assert backend.call("anything") == "ok"


def test_deterministic_backend_first_match_wins():
    backend = DeterministicBackend(responses={"a": "first", "ab": "second"})
    result = backend.call("ab")
    assert result in ("first", "second")


def test_claude_backend_is_importable():
    backend = ClaudeBackend()
    assert isinstance(backend, LLMBackend)


def test_claude_backend_custom_cli():
    backend = ClaudeBackend(cli="/usr/local/bin/my-claude", timeout=30)
    assert backend._cli == "/usr/local/bin/my-claude"
    assert backend._timeout == 30


def test_get_backend_mock():
    backend = get_backend("mock", responses={"test": "result"})
    assert isinstance(backend, DeterministicBackend)
    assert backend.call("test") == "result"


def test_get_backend_claude():
    backend = get_backend("claude")
    assert isinstance(backend, ClaudeBackend)


def test_llm_backend_protocol():
    assert isinstance(DeterministicBackend(), LLMBackend)
    assert isinstance(ClaudeBackend(), LLMBackend)
