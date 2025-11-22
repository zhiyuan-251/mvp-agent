import os
from typing import Dict, List, Optional

import requests


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url or self._infer_base_url(model)
        self.api_key_env = api_key_env or self._infer_api_key_env(self.base_url)
        self.api_key = api_key or os.getenv(self.api_key_env, "")
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                f"API key missing for model {model}. Set {self.api_key_env} or pass api_key."
            )

    @staticmethod
    def _infer_base_url(model: str) -> str:
        # Default to SiliconFlow for slash-separated model ids; otherwise OpenAI endpoint
        if "/" in model or model.lower().startswith(("deepseek", "qwen", "glm", "thudm")):
            return "https://api.siliconflow.cn/v1/chat/completions"
        return "https://api.openai.com/v1/chat/completions"

    @staticmethod
    def _infer_api_key_env(base_url: str) -> str:
        return "DEEPSEEK_API_KEY" if "siliconflow" in base_url else "OPENAI_API_KEY"

    @classmethod
    def from_model(cls, model: str, api_config: Optional[Dict] = None) -> "LLMClient":
        cfg = api_config or {}

        # 检查是否是OpenAI模型，如果是则使用OpenAI反代配置
        if model.startswith("gpt-") or model in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]:
            base_url = cfg.get("openai_base_url") or cfg.get("base_url")
            api_key = cfg.get("openai_api_key") or cfg.get("api_key")
            api_key_env = "OPENAI_API_KEY"  # 对于OpenAI模型使用固定的环境变量名
        else:
            base_url = cfg.get("base_url")
            api_key = cfg.get("api_key")
            api_key_env = cfg.get("api_key_env")

        timeout = cfg.get("timeout", 60) if isinstance(cfg, dict) else 60
        return cls(model=model, base_url=base_url, api_key=api_key, api_key_env=api_key_env, timeout=timeout)

    def complete(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
    ) -> str:
        # increment global call counter for cost estimation/metrics
        try:
            from utils import llm_client as _llm_mod
            _llm_mod._LLM_CALL_COUNT += 1
        except Exception:
            pass

        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(self.base_url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data and data["choices"]:
            choice = data["choices"][0]
            if "message" in choice and choice["message"].get("content"):
                return choice["message"]["content"].strip()
            if "text" in choice:
                return choice["text"].strip()
        raise RuntimeError(f"Unexpected response format: {data}")


# module-level counter accessible for experiments
_LLM_CALL_COUNT = 0


def get_and_reset_call_count() -> int:
    """Return number of LLM complete calls since last reset and reset counter."""
    global _LLM_CALL_COUNT
    try:
        cnt = _LLM_CALL_COUNT
        _LLM_CALL_COUNT = 0
        return cnt
    except Exception:
        return 0
