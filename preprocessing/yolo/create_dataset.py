import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from preprocessing.yolo.dataset import YoloDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("QGIS to yolo format converter")
    parser.add_argument("--tiled_tif_path", required=True, type=Path, help="E.g. ../../turret/processed/tifs")
    parser.add_argument("--tiled_label_path", required=True, type=Path, help="E.g. ../../turret/processed/bounding_boxes/tiled")
    parser.add_argument("--output_path", required=True, type=Path, help="E.g. ../../turret_output_processed")
    parser.add_argument('--patch_size', type=int, nargs=2, default=[640, 640], help='patch dimensions')
    parser.add_argument("--how_to_split", default="only_val")

    args = parser.parse_args()
    if args.how_to_split != "only_val":
        raise NotImplementedError(f"how_to_split implemented for only_val only")
    return args


def main(tiled_tif_path: Path, tiled_label_path: Path, output_path: Path, patch_size: list[int, int] = [640, 640], metadata_file: str = "metadata.csv"):

    yolo_dataset = YoloDataset(path=output_path)
    label_files = [e.stem for e in sorted(tiled_label_path.glob("*.txt"))]

    # Putting everything in "val" as default
    data = []
    for fname in tqdm(label_files):
        
        src_label = tiled_label_path / f"{fname}.txt"
        dst_label = yolo_dataset.path / yolo_dataset.label_dir / yolo_dataset.val_dir / f"{fname}.txt"
        
        with open(src_label) as f:
            s = f.read()
            s2 = s.replace(",", " ")
        
        with open(dst_label, "w") as f:
            f.write(s2)

        src_img = tiled_tif_path / f"{fname}.tif"
        dst_img = yolo_dataset.path / yolo_dataset.image_dir / yolo_dataset.val_dir / f"{fname}.jpg"
        im = Image.open(src_img).convert("RGB")
        im = im.resize((patch_size[0], patch_size[1]), Image.LANCZOS)
        im.save(dst_img)

        im_array = np.array(im)

        # TODO: Hardcode columns, these should be defined somewhere
        data.append({
            "img_path": dst_img,
            "label_path": dst_label,
            "tif_patch_path": src_img,
            "tif_orig_name": src_img.stem.split("_", 1)[1],
            "split": yolo_dataset.val_dir,
            "img_width": patch_size[0],
            "img_height": patch_size[1],
            "num_objects": s.count("\n"),
            "pct_black": (im_array == 0).sum() / im_array.size
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path / metadata_file, index=False)

    yolo_dataset.to_yaml()


if __name__ == "__main__":
    args = parse_args()
    main(tiled_tif_path=args.tiled_tif_path, tiled_label_path=args.tiled_label_path, output_path=args.output_path, patch_size=args.patch_size)
