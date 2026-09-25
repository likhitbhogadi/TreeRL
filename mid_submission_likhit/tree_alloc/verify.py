"""Binary answer checking: math_verify when installed, else a normalized boxed-string match."""
from __future__ import annotations

import re
from typing import Optional


def extract_boxed(text: str) -> Optional[str]:
    """Content of the last \\boxed{...}, with brace matching."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalize(ans: str) -> str:
    s = ans.strip().strip("$").rstrip(".")
    s = re.sub(r"\\(left|right|!|,|;|:)", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "")
    s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", "", s)
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


class Verifier:
    def __init__(self, backend: str = "auto"):
        self.mv = None
        if backend in ("auto", "math_verify"):
            try:
                import math_verify

                self.mv = math_verify
            except ImportError:
                if backend == "math_verify":
                    raise
        self.backend = "math_verify" if self.mv else "string"

    def __call__(self, response: str, gold: str) -> float:
        pred = extract_boxed(response)
        if pred is None:
            return 0.0
        if self.mv is not None:
            try:
                g = self.mv.parse(f"\\boxed{{{gold}}}")
                p = self.mv.parse(f"\\boxed{{{pred}}}")
                if g and p:
                    return float(bool(self.mv.verify(g, p)))
            except Exception:
                pass
        return float(normalize(pred) == normalize(gold))
