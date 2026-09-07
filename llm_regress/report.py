from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .scoring import CaseResult
from .regress import Verdict


class Reporter:
    def __init__(self, report_dir: str | Path):
        self.report_dir = Path(report_dir)

    def write(self, results: list[CaseResult], verdict: Verdict) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        out = self.report_dir / "report.json"
        payload = {
            "verdict": asdict(verdict),
            "cases": [
                {
                    "case_id": r.case_id,
                    "case_name": r.case_name,
                    "output": r.output,
                    "average": r.average,
                    "scores": [asdict(s) for s in r.scores],
                }
                for r in results
            ],
        }
        out.write_text(json.dumps(payload, indent=2))
        return out
