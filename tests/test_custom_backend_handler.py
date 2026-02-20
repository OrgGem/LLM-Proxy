# SPDX-License-Identifier: MIT

"""Tests for the custom backend handler."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app.config_manager import CustomBackendConfig, AuthConfig
from proxy_app.custom_backend_handler import (
    _build_auth_headers,
    _build_request_body,
    _extract_content,
    _resolve_path,
)


class TestBuildAuthHeaders:
    def test_none_auth(self):
        cfg = CustomBackendConfig(
            id="t", endpoint="https://x.com", auth=AuthConfig(type="none")
        )
        headers = _build_auth_headers(cfg)
        assert headers == {}

    def test_bearer_auth(self):
        cfg = CustomBackendConfig(
            id="t",
            endpoint="https://x.com",
            auth=AuthConfig(type="bearer", token="my-token"),
        )
        headers = _build_auth_headers(cfg)
        assert headers == {"Authorization": "Bearer my-token"}

    def test_api_key_auth(self):
        cfg = CustomBackendConfig(
            id="t",
            endpoint="https://x.com",
            auth=AuthConfig(
                type="api_key",
                token="key-123",
                header_name="X-API-Key",
                header_prefix="",
            ),
        )
        headers = _build_auth_headers(cfg)
        assert headers == {"X-API-Key": "key-123"}

    def test_oauth2_with_token(self):
        cfg = CustomBackendConfig(
            id="t",
            endpoint="https://x.com",
            auth=AuthConfig(
                type="oauth2_client_credentials",
                client_id="cid",
                client_secret="csecret",
                token_url="https://auth.example.com/token",
            ),
        )
        headers = _build_auth_headers(cfg, token="fetched-token")
        assert headers == {"Authorization": "Bearer fetched-token"}


class TestBuildRequestBody:
    def test_default_body(self):
        cfg = CustomBackendConfig(id="t", endpoint="https://x.com")
        messages = [{"role": "user", "content": "Hello"}]
        body = _build_request_body(cfg, messages, {"temperature": 0.5})
        assert body["messages"] == messages
        assert body["temperature"] == 0.5

    def test_prompt_template(self):
        cfg = CustomBackendConfig(
            id="t",
            endpoint="https://x.com",
            prompt_template='{"input": "$prompt"}',
        )
        messages = [
            {"role": "user", "content": "Hi there"},
        ]
        body = _build_request_body(cfg, messages, {})
        assert isinstance(body, dict)
        assert "user: Hi there" in body["input"]

    def test_prompt_template_invalid_json(self):
        cfg = CustomBackendConfig(
            id="t",
            endpoint="https://x.com",
            prompt_template="plain $prompt text",
        )
        messages = [{"role": "user", "content": "Test"}]
        body = _build_request_body(cfg, messages, {})
        # Should wrap in {"prompt": ...}
        assert "prompt" in body


class TestExtractContent:
    def test_openai_format(self):
        cfg = CustomBackendConfig(id="t", endpoint="https://x.com")
        data = {
            "choices": [
                {"message": {"content": "Hello world"}}
            ]
        }
        assert _extract_content(data, cfg) == "Hello world"

    def test_simple_result_field(self):
        cfg = CustomBackendConfig(id="t", endpoint="https://x.com")
        data = {"result": "Answer text"}
        assert _extract_content(data, cfg) == "Answer text"

    def test_response_mapping(self):
        cfg = CustomBackendConfig(
            id="t",
            endpoint="https://x.com",
            response_mapping={"content": "data.output"},
        )
        data = {"data": {"output": "Mapped content"}}
        assert _extract_content(data, cfg) == "Mapped content"

    def test_string_response(self):
        cfg = CustomBackendConfig(id="t", endpoint="https://x.com")
        assert _extract_content("plain text", cfg) == "plain text"


class TestResolvePath:
    def test_simple_key(self):
        assert _resolve_path({"a": "val"}, "a") == "val"

    def test_nested_key(self):
        assert _resolve_path({"a": {"b": "val"}}, "a.b") == "val"

    def test_list_index(self):
        assert _resolve_path({"a": [10, 20, 30]}, "a.[1]") == "20"

    def test_complex_path(self):
        data = {"choices": [{"message": {"content": "hi"}}]}
        assert _resolve_path(data, "choices.[0].message.content") == "hi"
