import sys
import warnings
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf, open_dict

from instanovo_marg.__init__ import console
from instanovo_marg.diffusion.train import train as train_diffusion
from instanovo_marg.transformer.train import _set_author_neptune_api_token
from instanovo_marg.utils import SpectrumDataFrame
from instanovo_marg.utils.colorlogging import ColorLog
from instanovo_marg.utils.marginal_distribution import get_marginal_distribution

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent / "configs"

warnings.filterwarnings("ignore", message=".*does not have many workers*")
_set_author_neptune_api_token()
logger = ColorLog(console, __name__).logger

DEFAULT_TRAIN_CONFIG_PATH = "instanovo_marg/configs"
DEFAULT_INFERENCE_CONFIG_PATH = "configs/inference"
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


config = compose_config(
    config_path=config_path,
    config_name=config_name,
    overrides=["model.dropout=0.1"],
)


if config["n_gpu"] > 1:
    raise ValueError("n_gpu > 1 currently not supported.")


logger.info("Initializing instanovo_marg+ training.")
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA version: {torch.version.cuda}")

# # Unnest hydra configs
# # TODO Use the nested configs by default
sub_configs_list = ["model", "dataset", "residues"]
for sub_name in sub_configs_list:
    if sub_name in config:
        with open_dict(config):
            temp = config[sub_name]
            del config[sub_name]
            config.update(temp)


logger.info(f"instanovo_marg+ training config:\n{OmegaConf.to_yaml(config)}")

with open(f"{config_path}/{config_name}.yaml", "r") as file:
    config_tmp = yaml.safe_load(file)


residues = config_tmp.get("defaults", {})[3]["residues"]
# sdf = 

# get_marginal_distribution(extended_vocab=True)

train_diffusion(config)
