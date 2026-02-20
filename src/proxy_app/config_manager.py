# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""
Custom backend config manager.

Handles loading, saving, caching and refreshing custom REST API backend
configurations from a JSON file. Each config entry defines how the proxy
should route requests when the requested model matches the config ID.
"""

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Pydantic Models ---

class AuthConfig(BaseModel):
    """Authentication configuration for a backend."""
    type: str = Field(
        default="none",
        description="Auth type: 'none', 'bearer', 'api_key', 'oauth2_client_credentials'",
    )
    token: Optional[str] = Field(default=None, description="Static bearer token or API key")
    header_name: Optional[str] = Field(
        default="Authorization",
        description="Header name for the token (default: Authorization)",
    )
    header_prefix: Optional[str] = Field(
        default="Bearer",
        description="Prefix before the token value (default: Bearer)",
    )
    client_id: Optional[str] = Field(default=None, description="OAuth2 client ID")
    client_secret: Optional[str] = Field(default=None, description="OAuth2 client secret")
    token_url: Optional[str] = Field(default=None, description="OAuth2 token endpoint URL")


class CustomBackendConfig(BaseModel):
    """Configuration for a custom REST API backend."""
    id: str = Field(description="Unique identifier used as the model name in requests")
    endpoint: str = Field(description="Backend REST API URL to forward requests to")
    auth: AuthConfig = Field(default_factory=AuthConfig, description="Authentication config")
    prompt_template: Optional[str] = Field(
        default=None,
        description=(
            "Template string to build the request body. "
            "Use {{messages}} for the messages JSON array and {{prompt}} for "
            "the concatenated message text."
        ),
    )
    method: str = Field(default="POST", description="HTTP method (default: POST)")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional headers to send with the request",
    )
    response_mapping: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Mapping from backend response fields to OpenAI response fields. "
            "Example: {\"result\": \"choices[0].message.content\"}"
        ),
    )
    description: Optional[str] = Field(default=None, description="Human-readable description")
    created_at: Optional[float] = Field(default=None, description="Creation timestamp")
    updated_at: Optional[float] = Field(default=None, description="Last update timestamp")


# --- Config Manager ---

class ConfigManager:
    """Manages custom backend configs stored in a JSON file."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self._config_path = Path(config_path)
        else:
            # Default: custom_configs.json next to this file's project root
            self._config_path = self._default_config_path()
        self._configs: Dict[str, CustomBackendConfig] = {}
        self._lock = threading.Lock()
        self._last_modified: float = 0.0
        self._load()

    @staticmethod
    def _default_config_path() -> Path:
        """Return the default config file path."""
        # Use the application root directory
        import sys

        if getattr(sys, "frozen", False):
            root = Path(sys.executable).parent
        else:
            root = Path.cwd()
        return root / "custom_configs.json"

    # --- Public API ---

    def get_all(self) -> List[CustomBackendConfig]:
        """Return all configs as a list."""
        self._maybe_reload()
        with self._lock:
            return list(self._configs.values())

    def get(self, config_id: str) -> Optional[CustomBackendConfig]:
        """Return a single config by ID, or None."""
        self._maybe_reload()
        with self._lock:
            return self._configs.get(config_id)

    def add(self, config: CustomBackendConfig) -> CustomBackendConfig:
        """Add a new config. Raises ValueError if ID already exists."""
        with self._lock:
            if config.id in self._configs:
                raise ValueError(f"Config with id '{config.id}' already exists")
            now = time.time()
            config.created_at = now
            config.updated_at = now
            self._configs[config.id] = config
            self._save()
        return config

    def update(self, config_id: str, data: Dict[str, Any]) -> CustomBackendConfig:
        """Update an existing config. Raises KeyError if not found."""
        with self._lock:
            if config_id not in self._configs:
                raise KeyError(f"Config with id '{config_id}' not found")
            existing = self._configs[config_id]
            update_dict = existing.model_dump()

            # Handle nested auth update
            if "auth" in data and isinstance(data["auth"], dict):
                auth_dict = update_dict.get("auth", {})
                auth_dict.update(data["auth"])
                data["auth"] = auth_dict

            update_dict.update(data)
            update_dict["id"] = config_id  # Prevent ID change
            update_dict["updated_at"] = time.time()
            update_dict["created_at"] = existing.created_at
            self._configs[config_id] = CustomBackendConfig(**update_dict)
            self._save()
        return self._configs[config_id]

    def delete(self, config_id: str) -> None:
        """Delete a config by ID. Raises KeyError if not found."""
        with self._lock:
            if config_id not in self._configs:
                raise KeyError(f"Config with id '{config_id}' not found")
            del self._configs[config_id]
            self._save()

    def has(self, config_id: str) -> bool:
        """Check whether a config with the given ID exists."""
        self._maybe_reload()
        with self._lock:
            return config_id in self._configs

    def reload(self) -> None:
        """Force reload configs from file."""
        self._load()

    # --- Internal ---

    def _load(self) -> None:
        """Load configs from the JSON file."""
        with self._lock:
            if not self._config_path.exists():
                self._configs = {}
                self._last_modified = 0.0
                return
            try:
                mtime = self._config_path.stat().st_mtime
                with open(self._config_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                configs: Dict[str, CustomBackendConfig] = {}
                entries = raw if isinstance(raw, list) else raw.get("configs", [])
                for entry in entries:
                    try:
                        cfg = CustomBackendConfig(**entry)
                        configs[cfg.id] = cfg
                    except Exception as exc:
                        logger.warning("Skipping invalid config entry: %s", exc)
                self._configs = configs
                self._last_modified = mtime
                logger.info(
                    "Loaded %d custom backend config(s) from %s",
                    len(configs),
                    self._config_path,
                )
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse config file %s: %s", self._config_path, exc)
            except Exception as exc:
                logger.error("Failed to load config file %s: %s", self._config_path, exc)

    def _save(self) -> None:
        """Persist current configs to the JSON file (must be called under lock)."""
        try:
            data = [cfg.model_dump() for cfg in self._configs.values()]
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._last_modified = self._config_path.stat().st_mtime
        except Exception as exc:
            logger.error("Failed to save config file %s: %s", self._config_path, exc)

    def _maybe_reload(self) -> None:
        """Reload the file if it has been modified on disk since last load."""
        try:
            if self._config_path.exists():
                mtime = self._config_path.stat().st_mtime
                if mtime > self._last_modified:
                    self._load()
        except Exception:
            pass
