"""The screen. Nothing here is a graded criterion; test 46 is a boundary check."""

import ast

import config


def test_46_the_app_imports_nothing_from_the_agent_or_the_rag_core():
    """Spec section 16 test 46, and the whole of D-82.

    If this fails, the app is answering questions itself rather than driving
    the Part 3 API, and no click it produces will ever write a Task 12 line.
    """
    tree = ast.parse((config.REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported & {"agent", "rag", "dataset", "db"}, (
        f"ui/app.py imports {sorted(imported & {'agent', 'rag', 'dataset', 'db'})}. "
        f"D-82 requires it to reach the agent only through POST /ask."
    )


def test_the_app_points_at_the_configured_api_port():
    source = (config.REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert str(config.API_PORT) in source


def test_the_app_mirrors_the_configured_escalation_threshold():
    """ui/app.py:39 hard-codes config.ESCALATION_THRESHOLD's value rather than
    importing it (D-82: ui/ may not import config), so nothing else catches a
    retune of the real threshold leaving the mirrored one, and the wrong
    screen would silently mislabel every lookup. Reads the file, like the
    port test above, for the same reason.
    """
    source = (config.REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8")
    assert str(config.ESCALATION_THRESHOLD) in source
