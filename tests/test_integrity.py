"""Systems must never see labels. This test is the enforcement."""

import ast

from membench.config import REPO_ROOT

FORBIDDEN_NAMES = {"answer", "ability", "question_type", "evidence_session_ids", "is_abstention"}
FORBIDDEN_IMPORTS = ("membench.judge", "membench.metrics")


def system_modules():
    return sorted((REPO_ROOT / "membench" / "systems").glob("*.py"))


def test_systems_do_not_import_judge_or_metrics():
    for path in system_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(FORBIDDEN_IMPORTS), f"{path.name} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(FORBIDDEN_IMPORTS), f"{path.name} imports {alias.name}"


def test_systems_never_read_label_attributes():
    for path in system_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                raise AssertionError(f"{path.name} reads label attribute .{node.attr}")


def test_system_answer_signature_takes_no_question_object():
    source = (REPO_ROOT / "membench" / "systems" / "base.py").read_text()
    assert "def answer(self, question: str, question_date: str)" in source
