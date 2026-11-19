"""Unit tests for configuration subsystem."""

from config import RAGConfig, get_config, reset_config


def test_default_config_properties():
    """Verify default configuration has expected paths and defaults."""
    cfg = RAGConfig()
    assert cfg.chunk_size == 500
    assert cfg.chunk_overlap == 50
    assert cfg.embedding_model == "text-embedding-3-small"
    assert cfg.chat_model == "gpt-4o-mini"
    assert cfg.temperature == 0.0
    assert cfg.top_k == 4
    assert cfg.max_query_length == 2000
    assert cfg.allow_dangerous_deserialization is True
    assert "You are a retrieval QA assistant." in cfg.prompt_template


def test_config_env_overrides(monkeypatch):
    """Verify configuration respects environment variable overrides."""
    monkeypatch.setenv("RAG_CHUNK_SIZE", "800")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "80")
    monkeypatch.setenv("RAG_TOP_K", "6")
    monkeypatch.setenv("RAG_CHAT_MODEL", "gpt-4o")
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")

    cfg = RAGConfig.from_env()
    assert cfg.chunk_size == 800
    assert cfg.chunk_overlap == 80
    assert cfg.top_k == 6
    assert cfg.chat_model == "gpt-4o"
    assert cfg.log_level == "DEBUG"


def test_singleton_config():
    """Verify get_config returns singleton instance and reset_config clears it."""
    reset_config()
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2

    reset_config()
    cfg3 = get_config()
    assert cfg3 is not None
