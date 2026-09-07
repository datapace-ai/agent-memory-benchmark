from membench.config import REPO_ROOT, load_models, load_systems


def test_load_models_returns_typed_config():
    cfg = load_models(REPO_ROOT / "configs" / "models.yaml")
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.answer_model == "qwen3:14b"
    assert cfg.num_ctx == 40960
    assert cfg.temperature == 0.0
    assert cfg.seeds == (11, 22, 33)


def test_load_systems_lists_phase_two_systems():
    systems = load_systems(REPO_ROOT / "configs" / "systems.yaml")
    assert [s.name for s in systems] == ["oracle", "window", "mem0", "langmem", "cognee"]
    assert [s.kind for s in systems[:2]] == ["context_window", "context_window"]
    assert systems[0].params["evidence_only"] is True
    assert systems[1].params["token_budget"] == 32000
    assert all(s.params["top_k"] == 10 for s in systems[2:])


def test_model_config_is_frozen():
    cfg = load_models(REPO_ROOT / "configs" / "models.yaml")
    try:
        cfg.num_ctx = 1
    except Exception as exc:
        assert "frozen" in str(type(exc)).lower() or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("ModelConfig should be immutable")
