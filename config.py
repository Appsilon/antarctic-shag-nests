import random
import numpy as np
from pathlib import Path
from PIL import Image
from pydantic import BaseSettings
from typing import Optional

# Set this to your conda environment containing qgis
conda_env_name = "nests"

# Required to load metadata of large images
Image.MAX_IMAGE_PIXELS = 1e9
file_path = Path(__file__).parent.resolve()

class Settings(BaseSettings):

    base_path: Path = file_path
    data_path: Path = base_path / "data"
    result_path: Path = base_path / "results"
    qgis_path: Optional[str] = f"/opt/conda/envs/{conda_env_name}/bin/qgis"
    qgis_plugin_path: Optional[str] = f"/opt/conda/envs/{conda_env_name}/lib/qgis/plugins"
    seed: int = 42


class Yolo6(BaseSettings):

    base_path: Path = file_path / "YOLOv6"
    train_save_path: Path = base_path / "runs" / "train"
    eval_save_path: Path = base_path / "runs" / "val"


# Initialise settings and set seeds
settings = Settings()
yolo6_settings = Yolo6()

random.seed(settings.seed)
np.random.seed(settings.seed)