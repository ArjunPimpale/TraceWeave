"""Tests for project configuration defaults and the environment template."""

from __future__ import annotations

from dotenv import dotenv_values

from pecs.config import PecsSettings


def test_env_example_is_a_valid_dotenv_template():
    values = dotenv_values(".env.example")

    assert '"""' not in values
    assert values["OLLAMA_BASE_URL"] == "http://localhost:11434"
    assert values["EXTRACTION_WORKERS"] == "6"
    assert "NEO4J_ENABLED" not in values


def test_settings_use_documented_default_values(monkeypatch):
    monkeypatch.delenv("EXTRACTION_WORKERS", raising=False)
    monkeypatch.delenv("NEO4J_ENABLED", raising=False)

    settings = PecsSettings(_env_file=None)

    assert settings.EXTRACTION_WORKERS == 4
    assert not hasattr(settings, "NEO4J_ENABLED")


def test_legacy_graph_environment_values_are_ignored(monkeypatch):
    monkeypatch.setenv("NEO4J_ENABLED", "true")
    assert not hasattr(PecsSettings(_env_file=None), "NEO4J_ENABLED")
