# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""
Config management API router.

Provides CRUD endpoints for custom backend configurations:
  GET    /configs        — list all configs
  POST   /configs        — add a new config
  PUT    /configs/{id}   — update an existing config
  DELETE /configs/{id}   — delete a config

All endpoints require proxy API key authentication.
Sensitive fields (tokens, secrets) are masked in GET responses.
"""

import logging
import os
import re
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from proxy_app.config_manager import ConfigManager, CustomBackendConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["configs"])

# --- Auth dependency (mirrors main.py verify_api_key) ---
_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def _verify_config_api_key(auth: str = Depends(_api_key_header)):
    """Verify proxy API key for config management endpoints."""
    proxy_key = os.getenv("PROXY_API_KEY")
    if not proxy_key:
        return auth  # Open access when PROXY_API_KEY is not set
    if not auth or auth != f"Bearer {proxy_key}":
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth


def get_config_manager(request: Request) -> ConfigManager:
    """Dependency to get the ConfigManager from app state."""
    return request.app.state.config_manager


# --- Helpers ---

_SENSITIVE_FIELDS = {"token", "client_secret"}


def _mask_sensitive(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of the config dict with sensitive auth fields masked."""
    result = dict(data)
    auth = result.get("auth")
    if isinstance(auth, dict):
        auth = dict(auth)
        for field in _SENSITIVE_FIELDS:
            val = auth.get(field)
            if val and isinstance(val, str) and len(val) > 0:
                if len(val) > 8:
                    auth[field] = f"{val[:4]}{'*' * (len(val) - 8)}{val[-4:]}"
                else:
                    auth[field] = "*" * len(val)
        result["auth"] = auth
    return result


# --- Response Models ---

class ConfigListResponse(BaseModel):
    configs: List[Dict[str, Any]]
    count: int


# --- Endpoints ---

@router.get("/configs", response_model=ConfigListResponse)
async def list_configs(
    manager: ConfigManager = Depends(get_config_manager),
    _=Depends(_verify_config_api_key),
):
    """Return all custom backend configs (sensitive fields masked)."""
    configs = manager.get_all()
    return ConfigListResponse(
        configs=[_mask_sensitive(c.model_dump()) for c in configs],
        count=len(configs),
    )


@router.post("/configs", status_code=201)
async def create_config(
    body: CustomBackendConfig,
    manager: ConfigManager = Depends(get_config_manager),
    _=Depends(_verify_config_api_key),
):
    """Add a new custom backend config."""
    try:
        created = manager.add(body)
        return created.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.put("/configs/{config_id}")
async def update_config(
    config_id: str,
    request: Request,
    manager: ConfigManager = Depends(get_config_manager),
    _=Depends(_verify_config_api_key),
):
    """Update an existing custom backend config."""
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    try:
        updated = manager.update(config_id, data)
        return updated.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.delete("/configs/{config_id}", status_code=204)
async def delete_config(
    config_id: str,
    manager: ConfigManager = Depends(get_config_manager),
    _=Depends(_verify_config_api_key),
):
    """Delete a custom backend config."""
    try:
        manager.delete(config_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
