"""Problem loading. Handles the field names of the datasets bundled in datasets/eval/
(MATH500: Question/Answer, OlympiadBench: question/final_answer, AMC: question/answer)
and Omni-MATH (problem/answer)."""
from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

Q_KEYS = ("problem", "question", "Question", "text", "prompt")
A_KEYS = ("answer", "Answer", "final_answer", "label", "golden_answer")


@dataclass
class Problem:
    id: str
    question: str
    answer: str
    raw: dict = field(default_factory=dict, repr=False)  # the original JSON row


def _answer(v) -> str:
    if isinstance(v, list):
        return str(v[0])
    s = str(v)
    if s.startswith("[") and s.endswith("]"):  # OlympiadBench stores "['2']"
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list) and lst:
                return str(lst[0])
        except (ValueError, SyntaxError):
            pass
    return s


def load_problems(path: str, limit: Optional[int] = None, ids: Optional[Sequence[str]] = None,
                  question_key: Optional[str] = None, answer_key: Optional[str] = None) -> List[Problem]:
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            qk = question_key or next(k for k in Q_KEYS if k in d)
            ak = answer_key or next(k for k in A_KEYS if k in d)
            pid = str(d.get("unique_id", d.get("id", i)))
            out.append(Problem(pid, d[qk], _answer(d[ak]), d))
    if ids is not None:
        keep = set(map(str, ids))
        out = [p for p in out if p.id in keep]
    return out[:limit] if limit else out
