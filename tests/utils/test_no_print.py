"""Ban bare print() and `from rich import print` in pipeline code.

ruff's T20 covers `print(...)` but is blind to the rich alias: `from rich import
print` rebinds the name, so a call site looks identical to the linter and gets a
pass. Four modules in this repository used to do exactly that. This walks the
AST instead of trusting the name.
"""

import ast

from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[2] / "src"
SOURCE_FILES = sorted(SRC.rglob("*.py"))


def _is_main_guard(node: ast.stmt) -> bool:
    """Whether `node` is `if __name__ == "__main__":`."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and any(
            isinstance(c, ast.Constant) and c.value == "__main__"
            for c in test.comparators
        )
    )


def _prints_outside_main(tree: ast.Module) -> list[int]:
    """Line numbers of print() calls not inside an `if __name__` block."""
    exempt: set[int] = set()
    for node in tree.body:
        if _is_main_guard(node):
            exempt.update(
                child.lineno for child in ast.walk(node) if hasattr(child, "lineno")
            )

    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
        and node.lineno not in exempt
    ]


def _rich_print_imports(tree: ast.Module) -> list[int]:
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "rich"
        and any(alias.name == "print" for alias in node.names)
    ]


def test_the_scan_actually_sees_files() -> None:
    """A glob that silently matched nothing would make both tests below vacuous."""
    assert len(SOURCE_FILES) > 10


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.name))
def test_no_bare_print(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = _prints_outside_main(tree)
    assert not offenders, (
        f"{path.relative_to(SRC.parent)} calls print() at line(s)"
        f" {offenders} -- use src.utils.logging.get_logger instead"
    )


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.name))
def test_no_rich_print_import(path: Path) -> None:
    tree = ast.parse(path.read_text(), filename=str(path))
    offenders = _rich_print_imports(tree)
    assert not offenders, (
        f"{path.relative_to(SRC.parent)} imports rich.print at line(s) {offenders}"
        " -- it shadows the builtin, so T20 cannot see the call sites,"
        " and its markup prints literally through a plain handler"
    )
