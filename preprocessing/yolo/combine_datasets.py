import argparse
import pandas as pd
import shutil
from pathlib import Path
from preprocessing.yolo.dataset import YoloDataset
from config import settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Combine datasets")
    parser.add_argument("--train_paths", required=True, type=lambda x: [d for d in x.split(',')])
    parser.add_argument("--val_paths", required=True, type=lambda x: [d for d in x.split(',')])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", required=True, type=str)

    return parser.parse_args()


def main(train_paths: list[Path], val_paths: list[Path], name: str, output: Path = settings.data_path):
    """
    Combine image datasets from different folders into train and validation sets for use in YOLOv6.
    Uses YoloDataset to create the expected directory layout that YOLOv6 expects (see YoloDataset).
    Copies image datasets to relevant folders for training and validation.
    
    Note: train_paths and val_paths assumed to have same directory structure as YoloDataset.

    Parameters
    ----------
    train_paths: Directories containing images/labels to be used for training.
    val_paths: Directories containing images/labels to be used for validation.
    output: Path where new dataset will be saved.
    name: Name of newly created dataset.
    """

    for path in train_paths + val_paths:
        assert path.exists()

    output_path = output / name

    if output_path.exists():
        
        while True:
            user_input = input("Dataset already exists, press Enter to overwrite, or 'x' return.")
            
            if user_input.lower() == '':
                break
            if user_input.lower() == 'x':
                return YoloDataset(path=output_path)
        
        shutil.rmtree(output_path)

    # Initialise dataset
    output_path.mkdir()
    dataset = YoloDataset(path=output_path)

    dfs = []

    for set_dir, dataset_paths in zip((dataset.train_dir, dataset.val_dir), (train_paths, val_paths)):

        for dset_path in dataset_paths:

            df = pd.read_csv(dset_path / YoloDataset.metadata_file)

            # TODO: Hardcode columns, these should be defined somewhere
            for subdir, metadata_col in zip((dataset.image_dir, dataset.label_dir), ("img_path", "label_path")):
            
                to_path = output_path / subdir / set_dir

                for _, row in df.iterrows():
                    
                    from_path = Path(row[metadata_col])
                    shutil.copy2(from_path, to_path)

                df[metadata_col] = df[metadata_col].apply(lambda x: to_path / Path(x).name)
            
            df["split"] = set_dir
            dfs.append(df)

    full_df = pd.concat(dfs).reset_index(drop=True)
    print(f"Total train: {full_df[full_df['split'] == 'train'].shape[0]}")
    print(f"Total val: {full_df[full_df['split'] == 'val'].shape[0]}")
    
    full_df.to_csv(output_path / YoloDataset.metadata_file, index=False)


    return dataset


if __name__ == "__main__":
    
    args = parse_args()
    main(**vars(args))
    
    # combine(["Shag", "Shag_extra"], ["Turret"], data_path=settings.data_path / "fulloverlap_1280" / "03_processed_1280")