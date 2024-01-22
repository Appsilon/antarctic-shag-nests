# Preprocessing

## Tiling

Before we can train new models or infer on new datasets, we have to prepare the data. Images come in the form of large geolocated TIFFs with corresponding shapefiles containing locations of nests. There are three key points in turning this into a dataset that can be trained/inferred on:

- Ensure each TIFF is correctly matched with a shapefile
- Cut each TIFF and shapefile pair into tiles that can be processed through YOLO
- Convert shapefile data from coordinates to pixels using TIFF tile metadata

This can be done using the [Dataset](dataset.py) architecture and QGIS. Once a dataset has been defined, running `dataset.process()` will cut the TIFF and shapefiles into a `raw` subdirectory (under the dataset output path) containing:

- `tif` - contains images of each patch for each map
- `bbox` - contains nest bounding boxes for each map
- `bbox_cut_to_patches_txt_dir` - contains nest bounding boxes for each patch (cut to patch boundaries!) for each map in YOLO format (normalized xywh, labels to YOLO model)
- `bbox_cut_to_patches_shp_dir` - contains nest bounding boxes for each patch (cut to patch boundaries!) for each map as shp file. We can visualise how bounding boxes were cut when they were located on patch contours.

It will also create a YOLO dataset that can be used directly in training/inference.

### Metadata

Once a dataset has been processed, it can be registered with the `metadata.yaml`. This allows datasets to be combined into larger datasets for YOLO training (including training/validation split) and provides an easy way to infer on multiple datasets.

The full process of creating a dataset, processing it and registering it to the metadata file looks like:

```python
from preprocessing.dataset import Dataset, Metdata

metadata = Metadata(...)
dataset = Dataset(...)
dataset.process()
metadata.add_dataset(dataset)
```

## Combining datasets

To created a combined dataset from multiple directories, run the `data_preparation/combine_datasets.py` specifying which dataset(s) should be used for training and validation. This will create a new YOLOv6 compliant dataset that can used for training a model.

Note that this can also be done with the `Metadata` architecture introduced earlier, specifically `metadata.combine_datasets(...)`.
