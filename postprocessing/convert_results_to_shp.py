import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PIL import Image
from PIL.TiffTags import TAGS
from shapely.geometry import Polygon
from tqdm import tqdm
from typing import Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("QGIS to yolo format converter")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--path_to_tifs", required=True)
    parser.add_argument("--border_threshold", default=0.0, type=float)
    parser.add_argument("--output", required=True)

    return parser.parse_args()


def pix2coord(row, col, top_left, dx, dy):
    # takes row, col for Array[row, col]
    # returns x_geographical, y_geographical
    return col * dx + top_left[0], top_left[1] - row * dy


def process_file_name(file_name: str) -> str:
    suffixes = [
        ("_orig", 1.00),
        ("_res20", 0.20),
        ("_res25", 0.25),
        ("_res50", 0.50),
        ("_new_res_20", 0.20),
        ("_new_res_25", 0.25),
        ("_new_res_50", 0.50),
    ]
    for suffix, scale in suffixes:
        if file_name.endswith(suffix):
            return file_name.removesuffix(suffix), scale
    return file_name, 1.0


def process(data: pd.DataFrame, border_threshold: float, path_to_imgs: Path, output_path: Path, filename: Optional[str] = "raw.shp"):
    
    data["coords"] = None
    data["geometry"] = None
    grouped = data.groupby("image_id")

    for group_name, grouped_df in tqdm(grouped):
        
        tif_dir, scale = process_file_name(group_name)
        
        with Image.open(path_to_imgs / f"{tif_dir}.tif") as img:
            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
            top_left = meta_dict["ModelTiepointTag"][3:5]
            pixel_size = meta_dict["ModelPixelScaleTag"][:2]
            patch_size = img.size

        for idx, item in grouped_df.iterrows():

            # cols and rows refer to Array!
            # Array[row, col], row moves along vertical axes, col along horizontal;
            # row_min, col_min are in the top left corner
            # row_max, col_max are in the bottom right corner
            # OUTPUT FROM YOLOv6 in the COCO format: bbox[row_min, col_min, width, height]
            col_min = (item["bbox"][0]) / scale
            row_min = (item["bbox"][1]) / scale
            col_max = (item["bbox"][0] + item["bbox"][2]) / scale
            row_max = (item["bbox"][1] + item["bbox"][3]) / scale

            # coords in geographical sense
            # x moves along horizontal axis, y - vertical.
            # Xmin, Ymin are now in the left BOTTOM corner
            # Xmax, Ymax are now in the right TOP corner
            # coords[xmin, ymin, xmax, ymax]
            x_min, y_min = pix2coord(row_max, col_min, top_left, pixel_size[0], pixel_size[1])
            x_max, y_max = pix2coord(row_min, col_max, top_left, pixel_size[0], pixel_size[1])

            # The following section filter predictions to close to patch borders
            tl_x, tl_y = top_left
            dx, dy = pixel_size
            if (
                x_min - tl_x < border_threshold
                or tl_x + (dx * patch_size[0]) - x_max < border_threshold
                or tl_y - y_max < border_threshold
                or y_min - (tl_y - (dy * patch_size[1])) < border_threshold
            ):
                continue

            data.at[idx, "coords"] = [x_min, y_min, x_max, y_max]

            lat_point_list = [x_min, x_max, x_max, x_min, x_min]
            lon_point_list = [y_min, y_min, y_max, y_max, y_min]
            data.at[idx, "geometry"] = Polygon(zip(lat_point_list, lon_point_list))

    data.dropna(subset="geometry", inplace=True)
    data_geo = gpd.GeoDataFrame(data, crs="epsg:32721", geometry="geometry")
    data_geo["coords"] = [",".join(map(str, l)) for l in data_geo["coords"]]
    data_geo["bbox"] = [",".join(map(str, l)) for l in data_geo["bbox"]]

    output_path.mkdir(exist_ok=True, parents=True)
    output_file = output_path / filename
    data_geo.to_file(filename=str(output_file), driver="ESRI Shapefile")

    return output_file


def main(args):
    path_to_imgs = Path(args.path_to_tifs)
    output_path = Path(args.output)
    border_threshold = args.border_threshold

    # Open the file and load the file
    data = pd.read_json(args.predictions)
   
    process(
        data=data, 
        border_threshold=border_threshold,
        path_to_imgs=path_to_imgs, 
        output_path=output_path
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
