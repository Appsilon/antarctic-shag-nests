import shutil
import subprocess
from pathlib import Path
from pydantic import BaseModel, validator
from typing import Optional
from config import settings, yolo6_settings


class TrainYolo6(BaseModel):

    name: str
    dataset_path: Path
    batch_size: int = 16
    config_path: Path = settings.base_path / "yolo_interface/v6/yolov6l_finetune_shags.py"
    device: int = 0
    epochs: int = 400
    eval_interval: int = 20
    heavy_eval_range: int = 50

    @validator("dataset_path")
    def point_to_yaml(cls, v):

        if not str(v).endswith("dataset.yaml"):
            v = v / "dataset.yaml"

        return v

    def generate_cmd(self):
        
        return [
            "python", 
            "tools/train.py", 
            "--batch", str(self.batch_size),
            "--conf", str(self.config_path),
            "--data", str(self.dataset_path),
            "--device", str(self.device),
            "--epochs", str(self.epochs),
            "--name", self.name,
            "--eval-interval", str(self.eval_interval),
            "--heavy-eval-range", str(self.heavy_eval_range)
        ]

    def get_result_path(self):

        return settings.result_path / self.name

    def run(self):

        result_path = self.get_result_path()

        if result_path.exists():
            shutil.rmtree(result_path)

        cmd = self.generate_cmd()
        subprocess.run(cmd, cwd=yolo6_settings.base_path)

        return result_path


class EvalYolo6(BaseModel):

    model_name: str
    dataset_path: Path
    batch_size: int = 32
    confidence_threshold: float = 0.03
    iou_threshold: float = 0.65
    device: int = 0
    save_name: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.save_name is None:
            self.save_name = self.get_save_name()

    @validator("dataset_path")
    def point_to_yaml(cls, v):

        if not str(v).endswith("dataset.yaml"):
            v = v / "dataset.yaml"

        return v

    def generate_cmd(self):
        
        return [
            "python", 
            "tools/eval.py", 
            "--data", str(self.dataset_path),
            "--batch-size", str(self.batch_size),
            "--weights", str(self.get_weight_path()),
            "--device", str(self.device),
            "--do_pr_metric", "True",
            "--conf-thres", str(self.confidence_threshold),
            "--iou-thres", str(self.iou_threshold),
            "--save_dir", f"runs/val/{self.model_name}",
            "--name", self.save_name,
        ]

    def get_save_name(self):

        return self.dataset_path.parent.name

    def get_result_path(self):

        return yolo6_settings.eval_save_path / self.model_name / self.save_name
    
    def get_weight_path(self):

        return yolo6_settings.train_save_path / self.model_name / "weights/best_ckpt.pt"

    def run(self, overwrite: bool = False):

        yolo_result_path = self.get_result_path()
        result_path = settings.result_path / yolo_result_path.relative_to(yolo6_settings.eval_save_path)
        
        if result_path.exists():
            
            if not overwrite and any(file.is_file() for file in result_path.rglob('*')):
                print(f"Results for dataset: {self.save_name} already exist in {result_path.relative_to(settings.result_path)}, " \
                      "set 'overwrite=True' to re-process")
            
                return result_path
            
            shutil.rmtree(result_path)

        if yolo_result_path.exists():
            shutil.rmtree(yolo_result_path)

        cmd = self.generate_cmd()
        print(f"Running inference model {self.model_name} on dataset {self.save_name}")
        subprocess.run(cmd, cwd=yolo6_settings.base_path)
        shutil.copytree(yolo_result_path, result_path)

        return result_path
