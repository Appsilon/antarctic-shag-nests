import geopandas as gpd
import numpy as np
import re
import yaml
from difflib import SequenceMatcher
from pathlib import Path, PurePath
from pydantic import BaseModel, DirectoryPath, FilePath
from typing import Optional, Sequence

from preprocessing.qgis import tile
from preprocessing.yolo import create_dataset, combine_datasets
from preprocessing.yolo.dataset import YoloDataset
from preprocessing.utils import DatasetFolderStructure
from utils.gen import map_nested_dict
from config import settings

class Dataset(BaseModel):

    """
    Dataset model required to process raw TIFF/shapefile data to YOLO compliant data
    
    Parameters
    ----------
    Required
    --------
    base_path: Location of dataset
    
    Optional
    --------
    image_ids: List of regex expressions that will be recursively globbed from path / image_dir. Default = *.tif
    folders: A DatasetFolderStructure object. If not provided, it will be created from base_path.
    images: List of images to process within path / image_dir. If not provided, requires if image_ids is not passed.
    shapefiles: List of shapefiles corresponding to each image. If not provided, the best matching shapefile under shapefile_path will be used
    target_resolution: Which resolution the images should be rescaled to. Default = original resolution
    name: Name of dataset, defaults to folder name
    split_col: For shapefiles containing nests on more than one location, specify a column to filter on
    split_id: An identifier to filter for within the split_col
    geodetic: Whether the shapefiles are in geodetic coordinates and therefore require a conversion
    """

    base_path: DirectoryPath
    image_ids: list[str] = ["*.tif"]
    folders: Optional[DatasetFolderStructure] = None
    images: Optional[list[FilePath]] = None
    shapefiles: Optional[list[FilePath]] = None
    target_resolution: Optional[float] = None
    name: Optional[str] = None
    split_col: Optional[str] = None
    split_id: Optional[str] = None
    geodetic: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not self.folders:
            # Creating the necessary folder structure
            self.folders = DatasetFolderStructure(path=self.base_path)

        if not self.images:
            self.images = self.get_images()

        if not self.shapefiles:
            self.shapefiles = self.get_shp_from_images()

        self.name = self.name or self.base_path.name

    def get_images(self):
        
        images = []
        for item in self.image_ids:
            images.extend(list((self.folders.raw_tif_path).rglob(item)))

        return images

    def get_shp_from_images(self):

        shapefiles = []
        for img in self.images:
            
            shp_path = get_ground_truth(gt_path=self.folders.raw_shapefile_path, image_id=img.stem)
            
            if shp_path is None:
                raise FileNotFoundError(f"No matching shapefile found for image: {img.stem}")
            
            shapefiles.append(shp_path)

        return shapefiles

    def process(self, overwrite: bool = False, *args, **kwargs):

        if  (
                not overwrite 
                and self.folders.tiled_tif_path.exists() 
                and any(file.is_file() for file in self.folders.tiled_tif_path.rglob('*'))
            ):
            
            print("Tiled output directory exists and is not empty, set 'overwrite=True' to re-process")
            return

        for img, shp in zip(self.images, self.shapefiles):
        
            tif_path, shp_path = Path(img), Path(shp)
            tile.main(tif_path=tif_path, shp_path=shp_path, output_path=self.base_path, target_res=self.target_resolution, *args, **kwargs)

    def create_yolo_dataset(self, overwrite: bool = False):

        yolo_image_path = self.folders.yolo_dataset_path / YoloDataset.image_dir

        if not overwrite and yolo_image_path.exists() and any(file.is_file() for file in yolo_image_path.rglob('*')):
            
            print("YOLO directory exists and is not empty, set 'overwrite=True' to re-process")
        else:
            print("Creating YOLO dataset")
            create_dataset.main(
                tiled_tif_path=self.folders.tiled_tif_path, 
                tiled_label_path=self.folders.tiled_bounding_box_txt_path, 
                output_path=self.folders.yolo_dataset_path, 
                metadata_file=YoloDataset.metadata_file
            )

    def load_shp_data(self, shp_file_path):
        """Load shapefile data for a given shapefile"""

        gt = gpd.read_file(shp_file_path)

        if self.split_col:
            gt = gt[gt[self.split_col].str.contains(self.split_id, case=False)]

        if self.geodetic:
            gt = gt.to_crs(32721)

        if "id" in gt:
            gt.drop(columns="id", inplace=True)

        # Assuming the preds file is points (centers of nests)
        gt = gt.loc[~gt.geometry.isna()]

        return gt


class MetaDataset:

    """
    Metadata class to register datasets to and allow them to be combined or easily iterated over
    All metadata paths are saved relative to settings.base_path to make them location independent
    """

    datasets: dict[str, Dataset] = {}

    def __init__(self, data_path: Path = settings.data_path, filename: str = "metadata.yaml", overwrite: bool = False) -> None:
        
        self.data_path = data_path
        self.filename = filename
        self.metadata_path = self.data_path / self.filename
        self.datasets = {}
        
        if self.metadata_path.exists():
            if overwrite:
                self.metadata_path.unlink()
            else:
                self.datasets = self.load_data()

    def load_data(self):

        with open(self.metadata_path, 'r') as file:
            data = yaml.safe_load(file)

        return self.init_datasets(data)

    def init_datasets(self, data):
        
        # TODO: Hacky way of resolving relative paths from metadata.yaml
        data = map_nested_dict(data, lambda x: settings.base_path / x if isinstance(x, (str, PurePath)) and x.startswith(f"{settings.data_path.name}/") else x)
        return {name: Dataset(**d) for name, d in data.items()} if data else {}

    def add_dataset(self, dataset: Dataset, overwrite: bool = False):

        if dataset.name in self.datasets and not overwrite:
            print(f"Dataset '{dataset.name}' already exists in: {self.filename}, to overwrite set 'overwrite=True'")
            return

        self.datasets[dataset.name] = dataset
        self.save()

    def process_datasets(self, overwrite: bool = False, *args, **kwargs):

        for name, dataset in self.datasets.items():
            print(f"Processing dataset: {name}")
            dataset.process(overwrite=overwrite, *args, **kwargs)

    def create_yolo_datasets(self, overwrite: bool = False, *args, **kwargs):

        for name, dataset in self.datasets.items():
            print(f"Creating YOLO dataset for: {name}")
            dataset.create_yolo_dataset(overwrite=overwrite, *args, **kwargs)

    def combine_yolo_datasets(self, train_keys: list[str], val_keys: list[str], name: str):

        train_paths = [self.datasets[k].folders.yolo_dataset_path for k in train_keys]
        val_paths = [self.datasets[k].folders.yolo_dataset_path for k in val_keys]
        return combine_datasets.main(train_paths, val_paths, name=name, output=self.data_path)

    def set_non_defaults(self, updates):
        
        """
        Update the metadata with non-default values from a nested dictionary.
        """
        def update_dict(d, u):
            for k, v in u.items():
                if k in d:
                    if isinstance(v, dict):
                        d[k] = update_dict(d.get(k, {}), v)
                    else:
                        d[k] = v
                else:
                    print(f"Key: '{k}' not recognized and will be skipped.")
            return d

        with open(self.metadata_path, 'r') as file:
            data = yaml.safe_load(file)

        data = update_dict(data, updates)
        self.datasets = self.init_datasets(data)
        self.save()

    def save(self):

        write_data = {name: dataset.dict(exclude={"folders": True}) for name, dataset in self.datasets.items()}
        
        # Saving paths as strings relative to settings.base_path
        write_data = map_nested_dict(write_data, lambda x: str(x.relative_to(settings.base_path)) if isinstance(x, PurePath) else x)

        with open(self.metadata_path, 'w') as file:
            yaml.safe_dump(write_data, file, default_flow_style=False)

    def add_from_data_folder(self, overwrite: bool = False, **kwargs):
        """
        Adding datasets from data directory to metadata.
        """

        data_path = self.metadata_path.parent

        for p in data_path.glob("*"):
            if p.is_dir():
                try:
                    print(f"Adding directory: {p}")
                    dataset = Dataset(base_path=p, **kwargs)
                    self.add_dataset(dataset, overwrite=overwrite)
                except AssertionError as e:
                    print(e, "... directory was not added")

    def get_dataset_keys(self) -> list[str]:
        
        return list(self.datasets.keys())
                    

def get_ground_truth(gt_path: Path,  image_id: str, shapefiles: Optional[Sequence[str]] = None) -> gpd.GeoDataFrame:
    """Get ground truth shapefile for a given dataset"""

    # Shapefiles either provided or all globbed under ground truth path
    if shapefiles is not None:
        shapefiles = [gt_path / f for f in shapefiles]
    else:
        shapefiles = [f for f in gt_path.rglob(
            "*.shp") if not str(f.name).startswith(".")]

    if len(shapefiles) == 1:
        matching = shapefiles[0]
    else:
        matching = find_matching_shapefile(shapefiles, image_id)

    gt_file_path = (gt_path / matching).with_suffix(".shp")
    print(f"For image: {image_id}, using shapefile: {gt_file_path.relative_to(settings.data_path)}")

    return gt_file_path


def find_matching_shapefile(shapefile_paths: list[Path], image_id: str) -> Path:
    """
    Find shapefile matching a given identifier.

    Notes: Multiple scenarios possible so the method is not the cleanest.
        Dates and pre/suffixes have always been provided in consistently separated format (underscore delimeter),
        so refactoring will be required if new delimiters are introduced.
    """

    date_match = re.search(r"\d{2,4}_\d{2}_\d{2,4}", image_id)

    # Including parent folder as some datasets have date subdirectories
    matching = [f for f in shapefile_paths if date_match[0]
        in " ".join([f.parent.name, f.name])] if date_match else []

    if len(matching) == 1:
        matching = matching[0]

    else:
        if len(matching) > 1:
            # Narrow down to date matches
            shapefile_paths = matching

        # Remove date if present and look exclusively at pre/suffixes
        dateless_image_id = image_id.replace(date_match[0], "") if date_match else image_id

        # Find best match
        match_id = np.argmax(list(map(lambda f: SequenceMatcher(None, a=f.stem, b=dateless_image_id).ratio(), shapefile_paths)))
        matching = shapefile_paths[match_id]

    return matching
