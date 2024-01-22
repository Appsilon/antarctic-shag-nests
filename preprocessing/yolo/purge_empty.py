import os
from preprocessing.yolo.dataset import YoloDataset
from config import settings

# Trial will only print files being kept/deleted, will not delete anything
TRIAL = True
DATASET_PATH = settings.data_path / "cleaned_up_processed/combined_cleaned_up_purged"


def main():
    """
    Remove images with no labels from a dataset, both label files and the corresponding image file will be deleted.
    Run in trial mode first to view how many images will be removed from dataset.
    """
    # TODO: Separate function with path as input. Option in meta-function to purge validation data also

    dataset = YoloDataset(path=DATASET_PATH)

    image_path = dataset.path / dataset.image_dir / dataset.train_dir
    label_path = dataset.path / dataset.label_dir / dataset.train_dir

    total_images, purged_images = 0, 0
    for label_file in label_path.glob("*.txt"):

        total_images += 1

        if os.path.getsize(label_file) > 0:
            print(f"Keeping {label_file.stem}")
        else:
            purged_images += 1
            print(f"Deleting {label_file.stem}")

            image_file = (image_path / label_file.stem).with_suffix(dataset.image_fmt)

            if not image_file.exists():
                raise FileNotFoundError(f"No matching image file for label file named: {label_file.stem}")

            if not TRIAL:
                label_file.unlink()
                image_file.unlink()
            
    print(f"Original images: {total_images}, remaining: {total_images - purged_images}")

    if TRIAL:
        print(f"NOTE: Run in trial mode, no files were deleted")

if __name__ == "__main__":

    main()