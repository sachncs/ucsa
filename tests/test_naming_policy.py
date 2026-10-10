"""Names are either public or fully private; never single-underscore.

A single leading underscore is a convention that the language does not
enforce, so it is neither public nor private. Every source file must use a
public name, or a double-underscore name (mangled inside a class). A bare `_`
is allowed as the conventional throwaway.
"""

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEMI_PRIVATE = re.compile(r"^_[A-Za-z0-9]")
# Standard-library APIs that happen to start with an underscore.
LIBRARY_NAMES = frozenset({"_exit", "_python_dispatch"})


def semi_private_names(source: str) -> list[tuple[int, str]]:
    """Returns `(line, name)` for every single-underscore identifier."""
    found = []
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            names.append(node.name)
        elif isinstance(node, ast.arg):
            names.append(node.arg)
        elif isinstance(node, ast.alias):
            names.append((node.asname or node.name).split(".")[-1])
        for name in names:
            if (
                SEMI_PRIVATE.match(name)
                and not name.startswith("__")
                and name not in LIBRARY_NAMES
            ):
                found.append((getattr(node, "lineno", 0), name))
    return sorted(set(found))


def test_detector_flags_semi_private_and_allows_the_rest():
    src = "def _a(): pass\nclass B:\n    def __c(self): pass\nx = _\n"
    assert [n for _, n in semi_private_names(src)] == ["_a"]


def repository_sources() -> list[pathlib.Path]:
    return sorted(
        p
        for folder in ("ucsa", "scripts", "tests")
        for p in (ROOT / folder).rglob("*.py")
        if "__pycache__" not in p.parts
    )


@pytest.mark.parametrize(
    "path", repository_sources(), ids=lambda p: str(p.relative_to(ROOT))
)
def test_the_whole_repository_has_no_semi_private_names(path):
    assert semi_private_names(path.read_text()) == [], path
