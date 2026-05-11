"""LLM client: Anthropic Messages API protocol with retry and JSON parsing."""

import json
import time
import logging

import requests

from .config import LLMConfig

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"


class LLMClient:
    """Unified LLM client for text and vision calls."""

    def __init__(self, llm_config: LLMConfig, retries: int = 3, backoff_base: int = 2, timeout: int = 180):
        self.config = llm_config
        self.retries = retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        if not llm_config.base_url:
            raise ValueError("LLM_BASE_URL is required")
        if not llm_config.api_key:
            raise ValueError("LLM_API_KEY is required")
        if not llm_config.model or not llm_config.vision_model:
            raise ValueError("LLM_MODEL and LLM_VISION_MODEL are required")

        self._endpoint = f"{llm_config.base_url.rstrip('/')}/v1/messages"
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        })

    def send_text(self, prompt: str, system: str | None = None) -> str:
        """Send a text-only prompt and return the response text."""
        messages = [{"role": "user", "content": prompt}]
        return self._call_api(messages, system, self.config.model)

    def send_image(self, image_b64: str, prompt: str, system: str | None = None) -> str:
        """Send an image + text prompt and return the response text."""
        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            },
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        return self._call_api(messages, system, self.config.vision_model)

    def send_messages(self, messages: list[dict], system: str | None = None) -> str:
        """Send pre-built messages and return the response text."""
        return self._call_api(messages, system, self.config.model)

    def _call_api(self, messages: list[dict], system: str | None, model: str) -> str:
        """Call the Anthropic Messages API with retry logic."""
        payload = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system

        for attempt in range(1, self.retries + 1):
            try:
                resp = self._session.post(
                    self._endpoint,
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()

                # Extract text from Anthropic response format
                content = result.get("content", [])
                text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
                return "\n".join(text_parts)

            except (requests.RequestException, TimeoutError, OSError) as e:
                logger.warning(f"API call failed (attempt {attempt}/{self.retries}): {e}")
                if attempt < self.retries:
                    wait = self.backoff_base ** attempt
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"API call failed after {self.retries} attempts: {e}")

        return ""

    @staticmethod
    def parse_json_response(text: str) -> dict | list | None:
        """Parse JSON from LLM response, handling markdown code blocks."""
        if not text:
            return None

        json_text = text.strip()

        # Strip markdown code block
        if "```" in json_text:
            start = json_text.index("```")
            if json_text[start:start + 7] == "```json":
                start += 7
            else:
                start += 3
            end = json_text.rfind("```")
            if end > start:
                json_text = json_text[start:end].strip()
            else:
                json_text = json_text[start:].strip()

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            # Try to find the first { or [ and last } or ]
            for open_char, close_char in [("{", "}"), ("[", "]")]:
                first = json_text.find(open_char)
                last = json_text.rfind(close_char)
                if first != -1 and last > first:
                    try:
                        return json.loads(json_text[first:last + 1])
                    except json.JSONDecodeError:
                        continue

            # Try to recover truncated JSON
            recovered = LLMClient._try_repair_json(json_text)
            if recovered is not None:
                return recovered

            logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
            return None

    @staticmethod
    def _try_repair_json(text: str) -> dict | list | None:
        """Attempt to repair truncated JSON by adding missing closing brackets."""
        # Count open vs close brackets/braces
        opens = text.count('{') + text.count('[')
        closes = text.count('}') + text.count(']')
        if opens <= closes:
            return None

        # Try adding missing closing characters
        suffix = ''
        stack = []
        for ch in text:
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch in ('}', ']'):
                if stack and stack[-1] == ch:
                    stack.pop()
        suffix = ''.join(reversed(stack))

        if suffix:
            try:
                return json.loads(text + suffix)
            except json.JSONDecodeError:
                # Try removing the last incomplete entry (e.g., truncated data point)
                # Find the last complete object/array entry
                for trim in range(1, min(200, len(text))):
                    trimmed = text[:-trim].rstrip().rstrip(',').rstrip()
                    attempt = trimmed + suffix
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        continue
        return None
