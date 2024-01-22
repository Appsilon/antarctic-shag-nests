# Training and Inference

## YOLOv6

[Helper functions](v6/model_configs.py) have been created to train models and infer on datasets which enable the full pipeline to be carried out from a single script or notebook. However, since these use Python subprocesses, it is likely they will run slower than from the command line.

In general it may be preferable to run inference using the helper as this is relatively short compared to training, and allows datasets to be processed and inferred on in a single script (as in the [quickstart](../end2end.ipynb)).

Below are instructions on how to carry out training/inference from the command line.

### Training

To train a model run `cd` to the location of your YOLOv6 repository and run the following command (see [YOLOv6 documentation](https://github.com/meituan/YOLOv6) for more):

```
python tools/train.py \
--batch 64 \
--conf configs/<python config file> \
--data <dataset path>/dataset.yaml \
--device 0
```

Where `dataset path` will point to a folder within the top level data folder of this repository (e.g. `../Antarctic-nests/data/combined_dataset`)

### Inference

Datasets to infer on require the same data preparation as discussed above. Then run the following command:

```
python tools/train.py \
--batch 64 \
--conf configs/<python config file> \
--data <dataset path>/dataset.yaml \
--device 0 \
--save_dir runs/val/<model name> \
--name <dataset name>
```

The above will save results to the YOLOv6 results folder. Either change this to the Antarctic-nests result folder, or copy after for processing.

Note that the `run_predictions` script will infer on all datasets and copy automatically.

### Common Errors

If you get `Exception: Dataset not found.` check paths in `dataset.yaml` file first.

## YOLOv8

### Training

Training with YOLOv8 requires the exact same version of data as with YOLOv6.
To run training run something similar to:

```sh
yolo detect train data=../data/february_experiments/shag_melville2_train_turret_val/dataset.yaml model=yolov8x.pt epochs=400 imgsz=640 batch=12
```