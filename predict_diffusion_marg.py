import glob
import sys
import warnings
from pathlib import Path
from typing import List, Optional

import lightning as L
import numpy as np
import torch
import typer
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.table import Table
from typing_extensions import Annotated

from instanovo_marg import __version__
from instanovo_marg.__init__ import console
from instanovo_marg.diffusion.multinomial_diffusion import InstaNovoPlus
from instanovo_marg.diffusion.predict import get_preds as diffusion_get_preds
from instanovo_marg.diffusion.train import train as train_diffusion
from instanovo_marg.scripts.convert_to_sdf import app as convert_to_sdf_app
from instanovo_marg.transformer.model import Instanovo_marg
from instanovo_marg.transformer.predict import get_preds as transformer_get_preds
from instanovo_marg.transformer.train import _set_author_neptune_api_token
from instanovo_marg.transformer.train import train as train_transformer
from instanovo_marg.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"

DEFAULT_TRAIN_CONFIG_PATH = "instanovo_marg/configs"
DEFAULT_INFERENCE_CONFIG_PATH = "instanovo_marg/configs/inference"
DEFAULT_INFERENCE_CONFIG_NAME = "default"
config_path = DEFAULT_TRAIN_CONFIG_PATH
config_name = "instanovoplus_marg"


def compose_config(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """Compose Hydra configuration with given overrides.

    Args:
        config_path: Relative path to config directory
        config_name: Name of the base config file
        overrides: List of Hydra override strings

    Returns:
        DictConfig: Composed configuration
    """
    logger.info(f"Reading config from '{config_path}' with name '{config_name}'.")
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides, return_hydra_config=False)
        return cfg


def run_diffusion_predict(
    data_path: str,
    output_path: str,
    instanovo_plus_model: str,
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    denovo: bool = False,
    refine: bool = False,
    overrides: Optional[List[str]] = None,
    transformer_prediction_path: str = None,
):
    """Python equivalent of the diffusion predict CLI command."""
    # Set default config paths if not provided
    if config_path is None:
        config_path = DEFAULT_INFERENCE_CONFIG_PATH
    if config_name is None:
        config_name = DEFAULT_INFERENCE_CONFIG_NAME

    # Compose the config (same as CLI)
    config = compose_config(config_path=config_path, config_name=config_name, overrides=overrides or [])

    # Set paths and parameters
    config.data_path = str(data_path)
    config.output_path = str(output_path)
    config.denovo = denovo
    config.refine = refine
    config.instanovo_plus_model = instanovo_plus_model
    config.instanovo_predictions_path = transformer_prediction_path

    # Validate paths
    if "*" in data_path and not glob.glob(data_path):
        raise ValueError(f"No files match pattern: {data_path}")
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    if instanovo_plus_model in InstaNovoPlus.get_pretrained():
        model, model_config = InstaNovoPlus.from_pretrained(instanovo_plus_model)
    else:
        model, model_config = InstaNovoPlus.load(instanovo_plus_model)

    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print("Configuration:\n", OmegaConf.to_yaml(config))

    # Save transformer predictions before refinement
    if config.get("refine", False) and not config.get("instanovo_predictions_path", ""):
        instanovo_predictions_path = Path(config.output_path).with_stem(
            Path(config.output_path).stem + "_before_refinement"
        )
        instanovo_predictions_path.write_text(Path(config.output_path).read_text())
        config.instanovo_predictions_path = str(instanovo_predictions_path)
    if config.get("refine", False) and config.get("instanovo_predictions_path", None) == config.get(
        "output_path", None
    ):
        raise ValueError(
            "The 'instanovo_predictions_path' should be different from the 'output_path' to avoid "
            "overwriting the transformer predictions."
        )

    # Run predictions
    diffusion_get_preds(config, model, model_config)

run_diffusion_predict(
    data_path="/home/seantang/projects/def-wan/seantang/InstaNovo/data/ms_proteometools/test.ipc",  # or *.mgf for multiple files
    output_path="marg_instanovo_plus_predictions.csv",
    instanovo_plus_model="checkpoints/instanovoplus-base/epoch_11_step_374999.ckpt",  # or pretrained ID
    config_path="instanovo_marg/configs/inference",
    config_name="default",
    denovo=False,
    refine=False,
    overrides=[]  # Optional overrides
)