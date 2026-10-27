"""Forge W&B: SDK uses api.*; browser run links use forge.*/wandb/..."""
from __future__ import annotations


def wandb_api_base_url(project: dict) -> str:
    return str(project.get("wandb_base_url") or "https://api.forge.coreweave.com").rstrip("/")


def wandb_ui_base_url(project: dict) -> str:
    return str(project.get("wandb_ui_base_url") or "https://forge.coreweave.com/wandb").rstrip("/")


def wandb_run_ui_url(project: dict, run_id: str) -> str:
    entity = project["wandb_entity"]
    name = project["wandb_project"]
    return f"{wandb_ui_base_url(project)}/{entity}/{name}/runs/{run_id}"
