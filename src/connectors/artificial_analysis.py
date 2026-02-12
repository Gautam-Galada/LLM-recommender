from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ArtificialAnalysisError(RuntimeError):
    """Raised when AA API call fails."""


@dataclass(slots=True)
class ArtificialAnalysisConnector:
    endpoint: str = "https://artificialanalysis.ai/api/v2/data/llms/models"

    def fetch(self) -> list[dict[str, Any]]:
        api_key = os.getenv("AA_API_KEY")
        if not api_key:
            raise ArtificialAnalysisError("AA_API_KEY is not set")

        request = Request(self.endpoint, headers={"x-api-key": api_key, "accept": "application/json"}, method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ArtificialAnalysisError(f"AA API HTTP error: {exc.code}") from exc
        except URLError as exc:
            raise ArtificialAnalysisError(f"AA API connection error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise ArtificialAnalysisError("AA API returned invalid JSON") from exc

        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = payload.get("data") or payload.get("models") or []
        else:
            records = []

        if not isinstance(records, list):
            raise ArtificialAnalysisError("AA API payload did not contain a model list")
        return [r for r in records if isinstance(r, dict)]
