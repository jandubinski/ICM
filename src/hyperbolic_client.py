# src/hyperbolic_client.py
import os
import time
import requests
from typing import List, Dict, Any, Optional, Union

HYPERBOLIC_CHAT_URL = "https://api.hyperbolic.xyz/v1/chat/completions"
HYPERBOLIC_COMP_URL = "https://api.hyperbolic.xyz/v1/completions"


class HyperbolicClient:
    """
    Hyperbolic API client supporting both chat (instruct) and raw completion endpoints.

    - .chat(...)      → for instruct/chat models
    - .complete(...)  → for base models (supports logprobs for Algorithm 1)
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key or os.getenv("HYPERBOLIC_API_KEY")
        if not self.api_key:
            raise ValueError("HYPERBOLIC_API_KEY not set")
        self.max_retries = max_retries

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # --------------------------
    # CHAT COMPLETIONS (Instruct models)
    # --------------------------
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int = 8,
        temperature: float = 0.0,
    ) -> str:
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        return self._post_with_retry(HYPERBOLIC_CHAT_URL, body)["choices"][0]["message"]["content"].strip()

    # --------------------------
    # RAW COMPLETIONS (Base models, logprobs supported)
    # --------------------------
    def complete(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 8,
        temperature: float = 0.0,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        return_json: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        If return_json=True or logprobs=True → returns full JSON (required for ICM Algorithm 1)
        Otherwise returns only the generated text.
        """
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if logprobs:
            body["logprobs"] = True
            if top_logprobs is not None:
                body["top_logprobs"] = top_logprobs

        data = self._post_with_retry(HYPERBOLIC_COMP_URL, body)

        if return_json or logprobs:
            return data  # caller will parse logprobs

        return data["choices"][0]["text"].strip()

    # --------------------------
    # Internal POST helper with retries
    # --------------------------
    def _post_with_retry(self, url: str, body: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                r = requests.post(url, headers=self.headers, json=body, timeout=60)

                if r.status_code == 200:
                    return r.json()

                if r.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue

                raise requests.HTTPError(f"{r.status_code} Error: {r.text}")

            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise e
                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError("Request failed after retries")
