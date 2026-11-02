"""Streaming reader over SocraticAI/knowledge.jsonl (370,231 records, 323 MB).

Never loads the file into memory. Records are yielded as plain dicts matching
schema v1.1 from SocraticAI/kb_schema.py; the 9,353 legacy schema-1.0 records
(no source_path, no license, origin "") are yielded too so Stage 1 of the filter
funnel can count and reject them explicitly rather than them vanishing silently.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..config import KNOWLEDGE_JSONL

FIELDS_V11 = (
    "id", "chunk_id", "schema_version", "origin", "source", "source_id",
    "source_path", "source_url", "license", "source_commit", "source_sha",
    "heading", "type", "subject", "predicate", "object", "claim",
    "confidence", "tags", "prompt_version", "model_version",
)


def iter_records(path: Path | None = None, limit: int | None = None) -> Iterator[dict]:
    """Yield every record. `limit` is for smoke tests only."""
    p = path or KNOWLEDGE_JSONL
    with open(p, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if limit is not None and n >= limit:
                return
            yield json.loads(line)


def doc_key(rec: dict) -> str:
    """The per-document key. source_path is authoritative; legacy records have
    none, so they fall back to `source` and are identifiable by that alone."""
    return rec.get("source_path") or rec.get("source") or ""


def group_key(rec: dict) -> tuple[str, str]:
    """The aggregation unit: 21,021 groups, median 17 records (plan §2.2)."""
    return (doc_key(rec), rec.get("heading") or "")


def is_legacy(rec: dict) -> bool:
    """Schema-1.0 shadow copy: no origin, no source_path, no license.
    Unattestable, and every one has a schema-1.1 twin."""
    return rec.get("origin") not in ("llm", "graph") or not rec.get("source_path")
