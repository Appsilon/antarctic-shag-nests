from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetFolderStructure:
    
    """
    Folder structure used for each individual dataset.

    Raw data should already be present within the dataset in the following form:

    path/
    ├── raw/
    │   ├── tifs/
    │   └── shapefiles/

    Folders will then be created to allow for processing.
    """

    path: Path

    @property
    def raw_path(self) -> Path:
        return self.path / "raw"
    
    @property
    def raw_tif_path(self) -> Path:
        return self.raw_path / "tifs"
    
    @property
    def raw_shapefile_path(self) -> Path:
        return self.raw_path / "shapefiles"
    
    @property
    def processed_path(self) -> Path:
        return self.path / "processed"
    
    @property
    def yolo_dataset_path(self) -> Path:
        return self.path / "yolo_dataset"
    
    @property
    def tiled_tif_path(self) -> Path:
        return self.processed_path / "tifs"

    @property
    def bounding_box_path(self) -> Path:
        return self.processed_path / "bounding_boxes"

    @property
    def original_bounding_box_path(self) -> Path:
        return self.bounding_box_path / "original"

    @property
    def tiled_bounding_box_path(self) -> Path:
        return self.bounding_box_path / "tiled" 
    
    @property
    def tiled_bounding_box_shp_path(self) -> Path:
        return self.tiled_bounding_box_path / "shp"
    
    @property
    def tiled_bounding_box_txt_path(self) -> Path:
        return self.tiled_bounding_box_path / "txt"
    
    def __init__(self, path: Path):
        
        self.path = path
        assert self.raw_path.exists(), f"Raw directory does not exist at: {self.raw_path}"
        assert self.raw_tif_path.exists(), f"Raw tif directory does not exist at: {self.raw_tif_path}"
        assert self.raw_shapefile_path.exists(), f"Shapefile directory does not exist at: {self.raw_shapefile_path}"

        self.create_subfolders()

    def create_subfolders(self):

        self.processed_path.mkdir(exist_ok=True)
        self.yolo_dataset_path.mkdir(exist_ok=True)
        self.tiled_tif_path.mkdir(exist_ok=True)
        self.bounding_box_path.mkdir(exist_ok=True)
        self.original_bounding_box_path.mkdir(exist_ok=True)
        self.tiled_bounding_box_path.mkdir(exist_ok=True)
        self.tiled_bounding_box_shp_path.mkdir(exist_ok=True)
        self.tiled_bounding_box_txt_path.mkdir(exist_ok=True)


@dataclass
class ResultsFolderStructure:
    
    """
    Folder structure used for storing results.
    """

    path: Path
    predictions_filename: str = "predictions.json"
    shapefile_folder: str = "shapefiles"
    raw_shapefile_name: str = "raw.shp"
    filtered_shapefile_name: str = "filtered.shp"
    processed_shapefile_name: str = "processed.shp"
    metrics_filename: str = "metrics.csv"

    @property
    def predictions_path(self) -> Path:
        return self.path / self.predictions_filename
    
    @property
    def shapefiles_path(self) -> Path:
        return self.path / self.shapefile_folder
    
    @property
    def raw_shapefile(self) -> Path:
        return self.shapefiles_path / self.raw_shapefile_name  

    @property
    def filtered_shapefile(self) -> Path:
        return self.shapefiles_path / self.filtered_shapefile_name   
    
    @property
    def processed_shapefile(self) -> Path:
        return self.shapefiles_path / self.processed_shapefile_name
    
    @property
    def metrics_file(self) -> Path:
        return self.path / self.metrics_filename

    def __init__(self, path: Path):
        
        self.path = path
        self.create_subfolders()

    def create_subfolders(self):

        self.path.mkdir(exist_ok=True)
        self.shapefiles_path.mkdir(exist_ok=True)