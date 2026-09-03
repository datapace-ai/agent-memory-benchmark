from membench.data.types import Question, Session
from membench.metrics.stats import bootstrap_ci, percentile
from membench.metrics.summary import summarize


def q(qid, ability, gap=0, n_sessions=4):
    sessions = tuple(
        Session(session_id=f"{qid}-s{i}", date="2023/05/01 (Mon) 10:00", order=i)
        for i in range(n_sessions)
    )
    evidence = {f"{qid}-s{n_sessions - 1 - gap}"}
    return Question(
        question_id=qid,
        question_type="multi-session",
        ability=ability,
        question="?",
        answer="a",
        question_date="2023/06/01 (Thu) 09:00",
        is_abstention=ability == "abstention",
        sessions=sessions,
        evidence_session_ids=frozenset(evidence),
    )


def rec(qid, system, seed, correct, **extra):
    base = {
        "question_id": qid, "system": system, "seed": seed, "answer_text": "x",
        "prompt_tokens": 1000, "completion_tokens": 20, "retrieval_seconds": 0.1,
        "answer_seconds": 1.0, "ingest_seconds": 2.0, "sessions_ingested": 4,
        "store_items": 4, "store_tokens": 400, "truncated": False, "error": None,
        "correct_longmemeval": correct, "correct_zep": correct, "correct_mem0": correct,
        "stale": None,
    }
    base.update(extra)
    return base


def test_percentile_matches_known_values():
    """Linear interpolation between ranks, matching numpy's default."""
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == 4.8
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([1, 2, 3, 4, 5], 0) == 1
    assert percentile([7], 95) == 7
    assert percentile([], 50) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1.0] * 8 + [0.0] * 2
    low, high = bootstrap_ci(values, iterations=500, seed=7)
    assert low <= 0.8 <= high
    assert bootstrap_ci(values, iterations=500, seed=7) == (low, high)


def test_summarize_computes_accuracy_per_system_and_ability():
    questions = {"a1": q("a1", "extraction"), "b1": q("b1", "temporal")}
    records = [
        rec("a1", "window", 11, True), rec("b1", "window", 11, False),
        rec("a1", "oracle", 11, True), rec("b1", "oracle", 11, True),
    ]
    out = summarize(records, questions)
    assert out["systems"]["window"]["accuracy"]["longmemeval"] == 0.5
    assert out["systems"]["oracle"]["accuracy"]["longmemeval"] == 1.0
    assert out["systems"]["window"]["by_ability"]["extraction"]["longmemeval"] == 1.0
    assert out["systems"]["window"]["by_ability"]["temporal"]["longmemeval"] == 0.0


def test_summarize_reports_share_of_oracle_and_gap_to_best_baseline():
    questions = {"a1": q("a1", "extraction"), "b1": q("b1", "temporal")}
    records = [
        rec("a1", "window", 11, True), rec("b1", "window", 11, False),
        rec("a1", "oracle", 11, True), rec("b1", "oracle", 11, True),
    ]
    out = summarize(records, questions)
    assert out["systems"]["window"]["share_of_oracle"] == 0.5
    assert out["systems"]["window"]["gap_to_best_baseline"] == 0.0


def test_summarize_flags_unstable_ranking_across_judges():
    questions = {"a1": q("a1", "extraction"), "b1": q("b1", "extraction")}
    records = [
        rec("a1", "window", 11, True, correct_zep=True, correct_mem0=False),
        rec("b1", "window", 11, False, correct_zep=True, correct_mem0=False),
        rec("a1", "other", 11, False, correct_zep=False, correct_mem0=True),
        rec("b1", "other", 11, True, correct_zep=False, correct_mem0=True),
    ]
    assert summarize(records, questions)["ranking_stable"] is False


def test_summarize_reports_latency_tokens_and_ingestion():
    questions = {"a1": q("a1", "extraction")}
    records = [rec("a1", "window", 11, True, answer_seconds=2.0, prompt_tokens=5000)]
    system = summarize(records, questions)["systems"]["window"]
    assert system["tokens_per_answer"] == 5020
    assert system["answer_latency_p50"] == 2.0
    assert system["ingest_seconds_per_session"] == 0.5


def test_summarize_counts_errors_as_wrong_and_reports_them():
    questions = {"a1": q("a1", "extraction"), "b1": q("b1", "extraction")}
    records = [rec("a1", "window", 11, True), rec("b1", "window", 11, False, error="boom")]
    out = summarize(records, questions)
    assert out["systems"]["window"]["accuracy"]["longmemeval"] == 0.5
    assert out["systems"]["window"]["errors"] == 1


def test_summarize_builds_a_forgetting_curve_by_evidence_gap():
    questions = {"a1": q("a1", "extraction", gap=0), "b1": q("b1", "extraction", gap=3)}
    records = [rec("a1", "window", 11, True), rec("b1", "window", 11, False)]
    curve = summarize(records, questions)["systems"]["window"]["forgetting_curve"]
    assert curve["0"] == 1.0
    assert curve["3"] == 0.0


def test_summarize_computes_stale_rate_for_knowledge_update():
    questions = {"k1": q("k1", "knowledge_update"), "k2": q("k2", "knowledge_update")}
    records = [
        rec("k1", "window", 11, False, stale=True),
        rec("k2", "window", 11, False, stale=False),
    ]
    assert summarize(records, questions)["systems"]["window"]["stale_rate"] == 0.5
