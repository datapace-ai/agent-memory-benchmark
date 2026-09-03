from membench.config import REPO_ROOT, load_models, load_systems


def test_load_models_returns_typed_config():
    cfg = load_models(REPO_ROOT / "configs" / "models.yaml")
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.answer_model == "qwen3:14b"
    assert cfg.num_ctx == 40960
    assert cfg.temperature == 0.0
    assert cfg.seeds == (11, 22, 33)


def test_load_systems_returns_two_phase_one_systems():
    systems = load_systems(REPO_ROOT / "configs" / "systems.yaml")
    assert [s.name for s in systems] == ["oracle", "window"]
    assert all(s.kind == "context_window" for s in systems)
    assert systems[0].params["evidence_only"] is True
    assert systems[1].params["token_budget"] == 32000


def test_model_config_is_frozen():
    cfg = load_models(REPO_ROOT / "configs" / "models.yaml")
    try:
        cfg.num_ctx = 1
    except Exception as exc:
        assert "frozen" in str(type(exc)).lower() or "cannot assign" in str(exc).lower()
    else:
        raise AssertionError("ModelConfig should be immutable")
