"""The two real tools the harness exercises: a calculator and a small fact lookup."""

import ast
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (+, -, *, /, **, parens). No eval()."""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree.body)
    return str(result)


_FACTS = {
    "days in a week": 7,
    "days in a year": 365,
    "hours in a day": 24,
    "minutes in an hour": 60,
    "weeks in a year": 52,
}


def lookup(query: str) -> str:
    """Look up a small fact from a fixed in-memory table (stub for a real search tool)."""
    key = query.strip().lower()
    if key in _FACTS:
        return str(_FACTS[key])
    matches = [k for k in _FACTS if key in k or k in key]
    if matches:
        return str(_FACTS[matches[0]])
    return f"NOT_FOUND: no fact matching '{query}'. Known facts: {', '.join(_FACTS)}"


CALCULATOR_SPEC = {
    "name": "calculator",
    "description": "Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses) and return the numeric result.",
    "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '(23 + 19) * 3'"}},
        "required": ["expression"],
    },
}

LOOKUP_SPEC = {
    "name": "lookup",
    "description": "Look up a small known fact (e.g. 'days in a week', 'hours in a day') from a reference table.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "the fact to look up"}},
        "required": ["query"],
    },
}
