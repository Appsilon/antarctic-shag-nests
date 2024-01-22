# antarctic-shag-nests

Experiments on the data recieved from prof. Robert Bialik - regarding identification of Cormoran nests in drone footage of an Antarctic island.

## Installation

It is recommended to create a new conda environment for this package, especially if you wish to work with QGIS (required for tiling images):

```bash
conda create -n nests python=3.10
conda activate nests
```

Install the project:

```bash
git clone https://github.com/Appsilon/Antarctic-nests.git
cd Antarctic-nests
pip install -r requirements.txt
```

### YOLOv6

From within the `Antartic-nests` directory, install YOLOv6 (or other YOLO versions):

```bash
git clone -b v3 https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```

Note that currently the repo works with the `v3` release of YOLOv6.

Both training and inference are carried out within the YOLOv6 `runs` directory, which are then copied over to the main repositories `results` directory.

During training the necessary folders will be created in the YOLOv6 directory, however for ad-hoc analysis an model folder will have to be placed in the `YOLOv6/runs/train/` directory with the following structure:

```bash
model_name
├── weights
|   └── best_ckpt.pt
```

During inference a `YOLOv6/runs/val/model_name` folder will be created with prediction json and weights file.

### QGIS (optional)

To prepare datasets from original TIFF and ground truth sets, QGIS also needs to be installed:

```bash
conda install qgis --channel conda-forge
```

To use QGIS the `conda_env_name` in `config.py` must be the name of the environment in which QGIS is located. If you installed QGIS in the `nests` environment you do not need to update this variable.

## Quickstart

The `end2end.ipynb` notebook contains all of the steps required to preprocess datasets (using QGIS), train and evaluate models, and post-process results.

For more information on the individiual steps and options, please see the following README's:

- [Preprocessing](preprocessing/README.md)
- [Training & Inference](yolo_interface/README.md)
- [Postprocessing](postprocessing/README.md)

## Datasets

The data used in this project can be found [here](https://console.cloud.google.com/storage/browser/antarctic-nests-data;tab=objects?forceOnBucketsSortingFiltering=false&project=wildlifeexplorer).

New datasets should be added to the data directory in the following format:

```bash
dataset_name
├── raw
|   ├── shapefiles
|   |   ├── shapefile1.shp
|   |   ├── shapefile2.shp
|   |   └── ...
|   ├── tifs
|   |   ├── image1.tif
|   |   ├── image2.tif
|   |   └── ...
```

### Multiple location shapefiles

In the case of multiple locations contained within a single shapefile, the dataset should be split into multiple datasets, each containing the TIFF files that correspond to that location and the shapefile.

The shapefile column and identifier that separates the locations must be provided in the `split_col` and `split_id` parameters, see the [end2end notebook](end2end.ipynb) for an example.