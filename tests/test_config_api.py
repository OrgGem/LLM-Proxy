# SPDX-License-Identifier: MIT

"""Tests for the config API endpoints."""

import json
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app.config_manager import ConfigManager
from proxy_app.config_api import router


@pytest.fixture
def app_client(tmp_path):
    """Create a test FastAPI app with config router."""
    app = FastAPI()
    config_path = str(tmp_path / "test_configs.json")
    manager = ConfigManager(config_path=config_path)
    app.state.config_manager = manager
    app.include_router(router)
    return TestClient(app)


class TestConfigAPI:
    def test_list_empty(self, app_client):
        resp = app_client.get("/configs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["configs"] == []
        assert data["count"] == 0

    def test_create_config(self, app_client):
        resp = app_client.post(
            "/configs",
            json={
                "id": "my_backend",
                "endpoint": "https://api.example.com/generate",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "my_backend"
        assert data["endpoint"] == "https://api.example.com/generate"

    def test_create_duplicate(self, app_client):
        payload = {"id": "dup", "endpoint": "https://example.com"}
        app_client.post("/configs", json=payload)
        resp = app_client.post("/configs", json=payload)
        assert resp.status_code == 409

    def test_list_after_create(self, app_client):
        app_client.post(
            "/configs",
            json={"id": "a", "endpoint": "https://a.com"},
        )
        app_client.post(
            "/configs",
            json={"id": "b", "endpoint": "https://b.com"},
        )
        resp = app_client.get("/configs")
        data = resp.json()
        assert data["count"] == 2

    def test_update_config(self, app_client):
        app_client.post(
            "/configs",
            json={"id": "upd", "endpoint": "https://old.com"},
        )
        resp = app_client.put(
            "/configs/upd",
            json={"endpoint": "https://new.com"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["endpoint"] == "https://new.com"
        assert data["id"] == "upd"

    def test_update_nonexistent(self, app_client):
        resp = app_client.put(
            "/configs/nope",
            json={"endpoint": "https://x.com"},
        )
        assert resp.status_code == 404

    def test_delete_config(self, app_client):
        app_client.post(
            "/configs",
            json={"id": "del_me", "endpoint": "https://x.com"},
        )
        resp = app_client.delete("/configs/del_me")
        assert resp.status_code == 204

        # Verify it's gone
        resp = app_client.get("/configs")
        assert resp.json()["count"] == 0

    def test_delete_nonexistent(self, app_client):
        resp = app_client.delete("/configs/nope")
        assert resp.status_code == 404

    def test_create_with_auth(self, app_client):
        resp = app_client.post(
            "/configs",
            json={
                "id": "auth_test",
                "endpoint": "https://api.example.com",
                "auth": {
                    "type": "bearer",
                    "token": "secret-token",
                },
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["auth"]["type"] == "bearer"
        assert data["auth"]["token"] == "secret-token"

    def test_create_with_prompt_template(self, app_client):
        resp = app_client.post(
            "/configs",
            json={
                "id": "tpl_test",
                "endpoint": "https://api.example.com",
                "prompt_template": '{"prompt": "$prompt"}',
            },
        )
        assert resp.status_code == 201
        assert resp.json()["prompt_template"] == '{"prompt": "$prompt"}'
