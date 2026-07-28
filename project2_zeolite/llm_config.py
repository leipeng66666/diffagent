"""
LLM API 配置文件
用户可根据实际使用的LLM服务修改此文件

支持两种模式：
1. Anthropic Claude API（推荐）
2. 兼容 OpenAI 接口的服务

使用方法：
    from llm_config import get_llm_client
    client = get_llm_client()
    response = client.chat("你的prompt")
"""

import os

# ============================================================
# 配置区域 - 请根据实际情况修改
# ============================================================

# 选择API提供商: "anthropic" 或 "openai"
API_PROVIDER = "openai"

# Anthropic 配置
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# OpenAI 兼容配置（DeepSeek）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")  # Set via .env file
OPENAI_BASE_URL = "https://api.deepseek.com"
OPENAI_MODEL = "deepseek-v4-pro"

# 通用配置
MAX_TOKENS = 8192
TEMPERATURE = 0.3  # Low temperature for analytical consistency


# ============================================================
# LLM 客户端封装
# ============================================================

class LLMClient:
    """统一的LLM调用接口"""

    def __init__(self, provider: str = "anthropic"):
        self.provider = provider
        self._client = None

    def _get_anthropic_client(self):
        """获取 Anthropic 客户端"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        except ImportError:
            raise ImportError(
                "请先安装 anthropic SDK: pip install anthropic"
            )

    def _get_openai_client(self):
        """获取 OpenAI 兼容客户端"""
        try:
            from openai import OpenAI
            return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        except ImportError:
            raise ImportError(
                "请先安装 openai SDK: pip install openai"
            )

    def chat(self, system_prompt: str, user_message: str,
             model: str = None, max_tokens: int = None,
             temperature: float = None) -> str:
        """
        发送聊天请求并返回文本响应

        Args:
            system_prompt: 系统提示词
            user_message: 用户消息
            model: 模型名称（可选，使用默认值）
            max_tokens: 最大token数（可选）
            temperature: 温度参数（可选）

        Returns:
            LLM的文本响应
        """
        if self.provider == "anthropic":
            return self._chat_anthropic(
                system_prompt, user_message, model, max_tokens, temperature
            )
        else:
            return self._chat_openai(
                system_prompt, user_message, model, max_tokens, temperature
            )

    def _chat_anthropic(self, system_prompt, user_message,
                        model, max_tokens, temperature):
        client = self._get_anthropic_client()
        model = model or ANTHROPIC_MODEL
        max_tokens = max_tokens or MAX_TOKENS
        temperature = temperature if temperature is not None else TEMPERATURE

        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return message.content[0].text

    def _chat_openai(self, system_prompt, user_message,
                     model, max_tokens, temperature):
        client = self._get_openai_client()
        model = model or OPENAI_MODEL
        max_tokens = max_tokens or MAX_TOKENS
        temperature = temperature if temperature is not None else TEMPERATURE

        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]
        )
        choice = response.choices[0]
        if choice.finish_reason == 'length':
            print(f"  [WARN] API: response truncated due to max_tokens limit ({max_tokens}) — consider increasing MAX_TOKENS")
        elif choice.finish_reason == 'stop':
            pass  # Normal completion
        else:
            print(f"  [WARN] API: unexpected finish_reason: {choice.finish_reason}")
        return choice.message.content


def get_llm_client(provider: str = None) -> LLMClient:
    """获取LLM客户端实例"""
    return LLMClient(provider=provider or API_PROVIDER)
