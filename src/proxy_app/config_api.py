# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel

"""
Config management API router.

Provides CRUD endpoints for custom backend configurations:
  GET    /configs        — list all configs
  POST   /configs        — add a new config
  PUT    /configs/{id}   — update an existing config
  DELETE /configs/{id}   — delete a config
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from proxy_app.config_manager import ConfigManager, CustomBackendConfig

logger = logging.getLogger(__name__)

router = APIRouter(tags=["configs"])


def get_config_manager(request: Request) -> ConfigManager:
    """Dependency to get the ConfigManager from app state."""
    return request.app.state.config_manager


# --- Response Models ---

class ConfigListResponse(BaseModel):
    configs: List[Dict[str, Any]]
    count: int


# --- Endpoints ---

@router.get("/configs", response_model=ConfigListResponse)
async def list_configs(
    manager: ConfigManager = Depends(get_config_manager),
):
    """Return all custom backend configs."""
    configs = manager.get_all()
    return ConfigListResponse(
        configs=[c.model_dump() for c in configs],
        count=len(configs),
    )


@router.post("/configs", status_code=201)
async def create_config(
    body: CustomBackendConfig,
    manager: ConfigManager = Depends(get_config_manager),
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
):
    """Delete a custom backend config."""
    try:
        manager.delete(config_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
