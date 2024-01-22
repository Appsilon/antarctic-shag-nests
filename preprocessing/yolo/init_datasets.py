from preprocessing.yolo.dataset import YoloDataset
from config import settings


def main():
    """
    Initialise all datasets within the data path.
    Recursively searches for directories with both an image_dir and label_dir (as defined by YoloDataset),
    creates the expected folder structure and a yaml file with the correct paths.
    """
    for d in settings.data_path.rglob("*"):

        if d.stem == "lost+found":
            continue
        
        if d.is_dir() and (d / YoloDataset.image_dir).exists() and (d / YoloDataset.label_dir).exists():
            YoloDataset(d, force_yaml=True)

if __name__ == "__main__":

    main()