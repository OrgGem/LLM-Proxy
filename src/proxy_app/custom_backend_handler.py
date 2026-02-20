# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""
Custom backend handler.

When the requested model matches a custom config ID, this module builds
the appropriate HTTP request, sends it to the configured backend, and
converts the response into the standard OpenAI chat completion format.
"""

import json
import logging
import time
import uuid
from string import Template
from typing import Any, Dict, Optional

import httpx

from proxy_app.config_manager import CustomBackendConfig

logger = logging.getLogger(__name__)

# Cache for OAuth2 client-credentials tokens
_token_cache: Dict[str, Dict[str, Any]] = {}


async def _get_oauth2_token(config: CustomBackendConfig) -> str:
    """Fetch an OAuth2 access token using client credentials, with caching."""
    cache_key = f"{config.auth.client_id}@{config.auth.token_url}"
    cached = _token_cache.get(cache_key)
    if cached and cached.get("expires_at", 0) > time.time() + 30:
        return cached["access_token"]

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            config.auth.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": config.auth.client_id,
                "client_secret": config.auth.client_secret,
            },
        )
        resp.raise_for_status()
        token_data = resp.json()

    access_token = token_data["access_token"]
    expires_in = token_data.get("expires_in", 3600)
    _token_cache[cache_key] = {
        "access_token": access_token,
        "expires_at": time.time() + expires_in,
    }
    return access_token


def _build_auth_headers(config: CustomBackendConfig, token: Optional[str] = None) -> Dict[str, str]:
    """Build authentication headers based on config."""
    headers: Dict[str, str] = {}
    auth = config.auth

    if auth.type == "none":
        return headers

    effective_token = token or auth.token or ""
    header_name = auth.header_name or "Authorization"
    prefix = auth.header_prefix or "Bearer"

    if auth.type in ("bearer", "oauth2_client_credentials"):
        headers[header_name] = f"{prefix} {effective_token}"
    elif auth.type == "api_key":
        headers[header_name] = effective_token

    return headers


def _build_request_body(
    config: CustomBackendConfig,
    messages: list,
    extra_params: Dict[str, Any],
) -> Any:
    """Build the request body to send to the backend."""
    if config.prompt_template:
        # Build a concatenated prompt string from messages
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content blocks
                text_parts = [
                    c.get("text", "") for c in content if isinstance(c, dict) and c.get("type") == "text"
                ]
                content = "\n".join(text_parts)
            prompt_parts.append(f"{role}: {content}")
        prompt_text = "\n".join(prompt_parts)

        messages_json = json.dumps(messages, ensure_ascii=False)

        try:
            # Use safe_substitute so missing keys don't raise
            tpl = Template(config.prompt_template)
            body_str = tpl.safe_substitute(
                messages=messages_json,
                prompt=prompt_text,
            )
            # Try parsing the result as JSON
            try:
                return json.loads(body_str)
            except json.JSONDecodeError:
                # If the template result isn't valid JSON, wrap in a dict
                return {"prompt": body_str}
        except Exception as exc:
            logger.warning("Failed to apply prompt_template, using default body: %s", exc)

    # Default body: forward messages as-is (OpenAI-compatible)
    body: Dict[str, Any] = {"messages": messages}
    # Forward commonly used parameters
    for key in ("temperature", "max_tokens", "top_p", "stop", "n"):
        if key in extra_params:
            body[key] = extra_params[key]
    return body


def _extract_content(response_data: Any, config: CustomBackendConfig) -> str:
    """Extract the assistant content from backend response using response_mapping or heuristics."""

    # If response_mapping is specified, use the defined content field path
    if config.response_mapping and "content" in config.response_mapping:
        path = config.response_mapping["content"]
        return _resolve_path(response_data, path)

    # Heuristic: try common response shapes
    if isinstance(response_data, dict):
        # OpenAI-compatible shape
        choices = response_data.get("choices")
        if isinstance(choices, list) and choices:
            msg = choices[0].get("message", {})
            if "content" in msg:
                return msg["content"] or ""
            delta = choices[0].get("delta", {})
            if "content" in delta:
                return delta["content"] or ""

        # Simple shapes
        for key in ("result", "response", "text", "output", "content", "answer", "data"):
            if key in response_data:
                val = response_data[key]
                if isinstance(val, str):
                    return val
                if isinstance(val, dict) and "text" in val:
                    return val["text"]

        # Fallback to full JSON
        return json.dumps(response_data, ensure_ascii=False)

    if isinstance(response_data, str):
        return response_data

    return str(response_data)


def _resolve_path(data: Any, path: str) -> str:
    """Resolve a dot-notation path (with optional list indexing) into a value."""
    parts = path.replace("[", ".[").split(".")
    current = data
    for part in parts:
        if not part:
            continue
        if part.startswith("[") and part.endswith("]"):
            idx = int(part[1:-1])
            current = current[idx]
        elif isinstance(current, dict):
            current = current.get(part, "")
        else:
            return ""
    return str(current) if current is not None else ""


async def handle_custom_backend(
    config: CustomBackendConfig,
    request_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Route a chat completion request to a custom backend and return
    an OpenAI-compatible response dict.
    """
    messages = request_data.get("messages", [])
    model = request_data.get("model", config.id)

    # Build auth
    token: Optional[str] = None
    if config.auth.type == "oauth2_client_credentials":
        token = await _get_oauth2_token(config)

    auth_headers = _build_auth_headers(config, token)

    # Build request
    body = _build_request_body(config, messages, request_data)
    headers = {"Content-Type": "application/json"}
    headers.update(config.headers)
    headers.update(auth_headers)

    logger.info("Routing request to custom backend '%s' → %s", config.id, config.endpoint)

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.request(
            method=config.method,
            url=config.endpoint,
            headers=headers,
            json=body if isinstance(body, (dict, list)) else None,
            content=body if isinstance(body, str) else None,
        )
        resp.raise_for_status()

    # Parse response
    try:
        response_data = resp.json()
    except Exception:
        response_data = resp.text

    content = _extract_content(response_data, config)

    # Build OpenAI-compatible response
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
