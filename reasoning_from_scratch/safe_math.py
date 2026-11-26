"""Restricted arithmetic parsing for untrusted generated answers."""

import io
import tokenize

import sympy
from sympy.parsing import sympy_parser as spp


def parse_math_expression(expression):
    """Reject Python strings, attributes and private names before SymPy eval."""
    if not isinstance(expression, str) or not expression or len(expression) > 2000:
        return None
    operators = {"+", "-", "*", "/", "**", "^", "%", "(", ")", ",", "[", "]"}
    try:
        for token in tokenize.generate_tokens(io.StringIO(expression).readline):
            if token.type == tokenize.NAME:
                if token.string.startswith("_") or "__" in token.string:
                    return None
            elif token.type == tokenize.OP:
                if token.string not in operators:
                    return None
            elif token.type not in (
                tokenize.NUMBER, tokenize.NEWLINE, tokenize.NL,
                tokenize.ENDMARKER, tokenize.ENCODING,
            ):
                return None
        # Include only arithmetic constructors/functions; Python builtins are
        # absent even after implicit-multiplication transformations run.
        names = (
            "Symbol", "Function", "Integer", "Float", "Rational",
            "sqrt", "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "log", "exp", "Abs", "factorial",
            "floor", "ceiling", "pi", "E", "I", "oo",
        )
        namespace = {name: getattr(sympy, name) for name in names}
        namespace["__builtins__"] = {}
        return spp.parse_expr(
            expression, global_dict=namespace,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (Exception, MemoryError):
        return None
