import json

from membench.report import main, render_markdown

SUMMARY = {
    "systems": {
        "oracle": {
            "questions": 100, "errors": 0, "truncated": 0,
            "accuracy": {"longmemeval": 0.84, "zep": 0.9, "mem0": 0.88},
            "accuracy_ci95": [0.78, 0.9],
            "by_ability": {"temporal": {"longmemeval": 0.8, "zep": 0.85, "mem0": 0.85}},
            "forgetting_curve": {"0": 0.9}, "stale_rate": 0.0, "tokens_per_answer": 4000.0,
            "answer_latency_p50": 3.0, "answer_latency_p95": 6.0,
            "retrieval_latency_p50": 0.01, "retrieval_latency_p95": 0.02,
            "ingest_seconds_per_session": 0.001, "store_tokens_final": 4000.0,
            "share_of_oracle": 1.0, "gap_to_best_baseline": 0.22,
        },
        "window": {
            "questions": 100, "errors": 2, "truncated": 1,
            "accuracy": {"longmemeval": 0.62, "zep": 0.7, "mem0": 0.68},
            "accuracy_ci95": [0.55, 0.69],
            "by_ability": {"temporal": {"longmemeval": 0.5, "zep": 0.6, "mem0": 0.6}},
            "forgetting_curve": {"0": 0.8, "3": 0.4}, "stale_rate": 0.35,
            "tokens_per_answer": 31000.0,
            "answer_latency_p50": 12.0, "answer_latency_p95": 20.0,
            "retrieval_latency_p50": 0.01, "retrieval_latency_p95": 0.02,
            "ingest_seconds_per_session": 0.001, "store_tokens_final": 40000.0,
            "share_of_oracle": 0.738, "gap_to_best_baseline": 0.0,
        },
    },
    "rankings": {"longmemeval": ["window"], "zep": ["window"], "mem0": ["window"]},
    "ranking_stable": True,
    "oracle_accuracy": 0.84,
    "best_baseline_accuracy": 0.62,
}


def test_render_markdown_has_the_leaderboard_rows():
    text = render_markdown(SUMMARY)
    assert "| window |" in text
    assert "| oracle |" in text
    assert "62.0" in text


def test_render_markdown_shows_all_three_judges():
    text = render_markdown(SUMMARY)
    assert "LongMemEval rule" in text and "Zep rule" in text and "Mem0 rule" in text


def test_render_markdown_marks_the_oracle_as_a_ceiling_not_a_competitor():
    assert "ceiling" in render_markdown(SUMMARY).lower()


def test_render_markdown_reports_errors_and_ranking_stability():
    text = render_markdown(SUMMARY)
    assert "Ranking is stable" in text
    assert "| 2 |" in text


def test_render_markdown_has_no_em_dashes():
    text = render_markdown(SUMMARY)
    assert "—" not in text and "–" not in text


def test_main_writes_summary_json_and_md(tmp_path):
    runs = tmp_path / "runs.jsonl"
    questions = tmp_path / "questions.jsonl"
    question = {
        "question_id": "q1", "question_type": "multi-session", "ability": "multi_session",
        "question": "?", "answer": "a", "question_date": "2023/06/01 (Thu) 09:00",
        "is_abstention": False, "evidence_session_ids": ["s0"],
        "sessions": [{"session_id": "s0", "date": "2023/05/01 (Mon) 10:00", "order": 0, "turns": []}],
    }
    questions.write_text(json.dumps(question) + "\n")
    record = {
        "question_id": "q1", "system": "window", "seed": 11, "answer_text": "a", "context": "",
        "prompt_tokens": 10, "completion_tokens": 2, "retrieval_seconds": 0.0,
        "answer_seconds": 1.0, "ingest_seconds": 0.5, "sessions_ingested": 1, "store_items": 1,
        "store_tokens": 10, "truncated": False, "error": None, "correct_longmemeval": True,
        "correct_zep": True, "correct_mem0": True, "stale": None,
    }
    runs.write_text(json.dumps(record) + "\n")

    out_dir = tmp_path / "out"
    assert main(["--runs", str(runs), "--questions", str(questions), "--out-dir", str(out_dir)]) == 0
    written = json.loads((out_dir / "summary.json").read_text())
    assert written["systems"]["window"]["accuracy"]["longmemeval"] == 1.0
    assert (out_dir / "summary.md").exists()
