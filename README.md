# Spot the shag!

This repo contains the source code for reproducing the results presented in the paper: [Using machine learning to count Antarctic shag (Leucocarbo bransfieldensis) nests on Remotely Piloted Aircraft Systems images
](url).

## Installation

It is recommended to create a new conda environment for this package, especially if you wish to work with QGIS (required for tiling images):

```bash
conda create -n nests python=3.10
conda activate nests
```

Install the project:

```bash
git clone https://github.com/Appsilon/antarctic-nests.git
cd antarctic-nests
pip install -r requirements.txt
```

### YOLOv6

From within the `antarctic-nests` directory, install YOLOv6:

```bash
git clone -b v3 https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```

Note that currently the repo works with the `v3` release of YOLOv6.

Both training and inference are carried out within the YOLOv6 `runs` directory, which are then copied over to the main repositories `results` directory.

An additional weights file must be downloaded in order to carry out training, which YOLO will attempt to download automatically. However, if this does not work, create a `YOLOv6/weights` directory, download the weights file using [this link](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6l.pt), and place it into the folder.

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

## Data

The data used in this project can be found [here](https://console.cloud.google.com/storage/browser/_details/antarctic-nests-data/all_islands.zip?project=wildlifeexplorer). Download the `.zip` archive and unzip it into the `data` folder. You can use the following command for the download (assuming you are in the `antarctic-nests/data` directory):

```bash
gsutil cp gs://antarctic-nests-data/all_islands.zip
```

## Quickstart

The `end2end.ipynb` notebook contains all of the steps required to preprocess datasets (using QGIS), train and evaluate models, and post-process results.

For more information on the individiual steps and options, please see the following README's:

- [Preprocessing](preprocessing/README.md)
- [Training & Inference](yolo_interface/README.md)
- [Postprocessing](postprocessing/README.md)

## Adding new datasets

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

### Pre-trained weights

In the [release](https://github.com/Appsilon/antarctic-nests/releases/tag/model) you will find the weights for the model results from which we published in the paper. To make use of them, download the attached `.zip` file and unzip into `YOLOv6/runs/train/...`.
