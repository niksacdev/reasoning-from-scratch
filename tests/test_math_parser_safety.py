"""Generated answers must never invoke Python code while being graded."""

import pytest

from reasoning_from_scratch.ch03 import sympy_parser
from reasoning_from_scratch.bonus.parser import sympy_parser_hybrid


@pytest.mark.parametrize("parse", [sympy_parser, sympy_parser_hybrid])
def test_python_payload_cannot_create_file(parse, tmp_path):
    marker = tmp_path / "must-not-exist"
    payload = f"__import__('pathlib').Path('{marker}').write_text('executed')"
    parse(payload)
    assert not marker.exists()


@pytest.mark.parametrize("payload", [
    "Symbol.__class__", "(1).__class__", "__builtins__", "open('/tmp/file','w')",
])
def test_python_syntax_is_not_arithmetic(payload):
    assert sympy_parser(payload) is None


@pytest.mark.parametrize("expression, expected", [
    ("2x + 3x", "5*x"), ("sqrt(4)", "2"), ("sin(pi/2)", "1"),
    ("1.25 + 0.75", "2.00000000000000"),
])
def test_arithmetic_and_implicit_multiplication_remain_supported(expression, expected):
    assert str(sympy_parser(expression)) == expected
