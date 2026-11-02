"""Minimal YAML emitter producing catalog-style output.

sekura-attack-trees ships yamllite, which only *parses*. Generating training
targets and RL candidates needs the other direction, and the output must
round-trip back through yamllite -- otherwise a generated tree cannot be scored
by the very validator that makes this data valuable.

Deliberately narrow: it emits only the shapes the attack-subtree schema uses
(nested mappings, lists of scalars, lists of mappings, ints, strings, empty
lists). It is not a general YAML writer and should not become one.
"""

from __future__ import annotations

import re

# Quote anything that could be misread as another YAML type, or that carries
# structural characters. The catalog quotes all its strings, so matching that
# keeps generated output visually identical to hand-authored files.
_PLAIN_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-/()]*$")


def _scalar(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if v is None:
        return '""'
    s = str(v)
    if "\n" in s:
        # Fold to one line: the schema has no multi-line string fields, and a
        # literal block would not survive yamllite's subset.
        s = " ".join(s.split())
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def dump(obj, indent: int = 0) -> str:
    """Serialize a mapping to catalog-style YAML."""
    pad = "  " * indent
    lines: list[str] = []
    for key, val in obj.items():
        if isinstance(val, dict):
            if not val:
                lines.append(f"{pad}{key}: {{}}")
            else:
                lines.append(f"{pad}{key}:")
                lines.append(dump(val, indent + 1))
        elif isinstance(val, list):
            if not val:
                lines.append(f"{pad}{key}: []")
                continue
            lines.append(f"{pad}{key}:")
            for item in val:
                if isinstance(item, dict):
                    body = dump(item, indent + 2).split("\n")
                    first = body[0].lstrip()
                    lines.append(f"{pad}  - {first}")
                    lines.extend(body[1:])
                else:
                    lines.append(f"{pad}  - {_scalar(item)}")
        else:
            lines.append(f"{pad}{key}: {_scalar(val)}")
    return "\n".join(lines)


def dumps(doc: dict) -> str:
    return dump(doc, 0) + "\n"
