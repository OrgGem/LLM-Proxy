# SPDX-License-Identifier: MIT

"""Tests for the ConfigManager and Config API."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proxy_app.config_manager import ConfigManager, CustomBackendConfig, AuthConfig


# --- ConfigManager unit tests ---


class TestConfigManager:
    """Tests for ConfigManager CRUD operations."""

    def _make_manager(self, tmp_path):
        config_path = str(tmp_path / "test_configs.json")
        return ConfigManager(config_path=config_path), config_path

    def test_empty_initial_load(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        assert manager.get_all() == []

    def test_add_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        cfg = CustomBackendConfig(
            id="test_backend",
            endpoint="https://api.example.com/v1/generate",
        )
        result = manager.add(cfg)
        assert result.id == "test_backend"
        assert result.endpoint == "https://api.example.com/v1/generate"
        assert result.created_at is not None
        assert result.updated_at is not None

    def test_add_duplicate_raises(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        cfg = CustomBackendConfig(id="dup", endpoint="https://example.com")
        manager.add(cfg)
        with pytest.raises(ValueError, match="already exists"):
            manager.add(cfg)

    def test_get_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        cfg = CustomBackendConfig(id="get_test", endpoint="https://example.com")
        manager.add(cfg)
        result = manager.get("get_test")
        assert result is not None
        assert result.id == "get_test"

    def test_get_nonexistent(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        assert manager.get("nope") is None

    def test_has_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.add(CustomBackendConfig(id="exists", endpoint="https://example.com"))
        assert manager.has("exists") is True
        assert manager.has("nope") is False

    def test_update_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.add(
            CustomBackendConfig(id="upd", endpoint="https://old.example.com")
        )
        updated = manager.update("upd", {"endpoint": "https://new.example.com"})
        assert updated.endpoint == "https://new.example.com"
        assert updated.id == "upd"  # ID must not change

    def test_update_nonexistent_raises(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            manager.update("no_such", {"endpoint": "https://x.com"})

    def test_delete_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.add(CustomBackendConfig(id="del_me", endpoint="https://example.com"))
        manager.delete("del_me")
        assert manager.get("del_me") is None

    def test_delete_nonexistent_raises(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        with pytest.raises(KeyError, match="not found"):
            manager.delete("no_such")

    def test_persistence(self, tmp_path):
        """Configs survive across manager instances."""
        config_path = str(tmp_path / "persist.json")
        m1 = ConfigManager(config_path=config_path)
        m1.add(
            CustomBackendConfig(
                id="persist_test",
                endpoint="https://example.com",
                description="test persistence",
            )
        )
        # Create new manager pointing to same file
        m2 = ConfigManager(config_path=config_path)
        result = m2.get("persist_test")
        assert result is not None
        assert result.description == "test persistence"

    def test_auth_config(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        cfg = CustomBackendConfig(
            id="auth_test",
            endpoint="https://api.example.com",
            auth=AuthConfig(
                type="bearer",
                token="test-token-123",
            ),
        )
        manager.add(cfg)
        result = manager.get("auth_test")
        assert result.auth.type == "bearer"
        assert result.auth.token == "test-token-123"

    def test_update_nested_auth(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.add(
            CustomBackendConfig(
                id="nested",
                endpoint="https://example.com",
                auth=AuthConfig(type="bearer", token="old-token"),
            )
        )
        updated = manager.update(
            "nested", {"auth": {"token": "new-token"}}
        )
        assert updated.auth.token == "new-token"
        assert updated.auth.type == "bearer"  # Preserved from original

    def test_get_all(self, tmp_path):
        manager, _ = self._make_manager(tmp_path)
        manager.add(CustomBackendConfig(id="a", endpoint="https://a.com"))
        manager.add(CustomBackendConfig(id="b", endpoint="https://b.com"))
        configs = manager.get_all()
        assert len(configs) == 2
        ids = {c.id for c in configs}
        assert ids == {"a", "b"}

    def test_reload_from_file(self, tmp_path):
        """Test that reload picks up external file changes."""
        config_path = str(tmp_path / "reload.json")
        manager = ConfigManager(config_path=config_path)
        manager.add(CustomBackendConfig(id="original", endpoint="https://o.com"))

        # Externally modify the file
        with open(config_path, "w") as f:
            json.dump(
                [
                    {"id": "external", "endpoint": "https://e.com"},
                ],
                f,
            )
        manager.reload()
        assert manager.get("external") is not None
        assert manager.get("original") is None
