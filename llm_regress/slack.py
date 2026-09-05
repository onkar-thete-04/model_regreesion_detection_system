from __future__ import annotations

import json
import os

import requests

from .regress import Verdict


class SlackNotifier:
    def __init__(self, webhook_url_env: str = "SLACK_WEBHOOK_URL"):
        self.webhook_url_env = webhook_url_env

    def send(self, feature: str, pr: str, run_url: str, verdict: Verdict) -> None:
        url = os.getenv(self.webhook_url_env)
        if not url:
            return
        text = (
            f":warning: *LLM regression detected*\n"
            f"feature: {feature}\n"
            f"PR: {pr}\n"
            f"aggregate: {verdict.aggregate_score:.2f}\n"
            f"baseline: {verdict.baseline_score if verdict.baseline_score is not None else 'n/a'}\n"
            f"run: {run_url}"
        )
        requests.post(url, json={"text": text}, timeout=10)
