from membench.config import REPO_ROOT, load_models, load_systems


def test_load_models_returns_typed_config():
    cfg = load_models(REPO_ROOT / "configs" / "models.yaml")
    assert cfg.provider == "openai_compat"
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.answer_model == "nvidia/nemotron-3.5-lightning:free"
    assert cfg.openai_compat_model == "nvidia/nemotron-3.5-lightning:free"
    assert cfg.num_ctx == 40960
    assert cfg.temperature == 0.0
    assert cfg.seeds == (11, 22, 33)
    assert cfg.think is False
    assert cfg.min_request_interval_seconds == 3.0
    assert cfg.embed_model == "BAAI/bge-small-en-v1.5" and cfg.embed_dims == 384


def test_ollama_config_file_still_loads():
    cfg = load_models(REPO_ROOT / "configs" / "models.ollama.yaml")
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.openai_compat_model == "qwen3-nothink:14b"
    assert cfg.min_request_interval_seconds == 0.0


def test_load_systems_lists_phase_two_systems():
    systems = load_systems(REPO_ROOT / "configs" / "systems.yaml")
    assert [s.name for s in systems] == ["oracle", "window", "file", "mem0", "langmem", "cognee", "graphiti", "letta"]
    assert [s.kind for s in systems[:2]] == ["context_window", "context_window"]
    assert systems[0].params["evidence_only"] is True
    assert systems[1].params["token_budget"] == 32000
    assert systems[2].kind == "file_search"
    assert all(s.params["top_k"] == 10 for s in systems[3:7])
    assert systems[7].kind == "letta"


def test_model_config_is_frozen():
    cfg = load_models(REPO_ROOT / "configs" / "models.yaml")
    try:
        cfg.num_ctx = 1
    except Exception as exc:
        assert "frozen" in str(type(exc)).lower() or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("ModelConfig should be immutable")
