from vn_tsd.runtime.wandb_urls import wandb_api_base_url, wandb_run_ui_url, wandb_ui_base_url


def test_forge_ui_run_url():
    project = {
        "wandb_entity": "HCMUT_IPCV",
        "wandb_project": "BTL",
        "wandb_base_url": "https://api.forge.coreweave.com",
        "wandb_ui_base_url": "https://forge.coreweave.com/wandb",
    }
    assert wandb_api_base_url(project) == "https://api.forge.coreweave.com"
    assert wandb_ui_base_url(project) == "https://forge.coreweave.com/wandb"
    assert (
        wandb_run_ui_url(project, "abc123")
        == "https://forge.coreweave.com/wandb/HCMUT_IPCV/BTL/runs/abc123"
    )
