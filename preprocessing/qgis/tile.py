import os
import argparse
from abc import ABC
from pathlib import Path
from PIL import Image
from PIL.TiffTags import TAGS
from tqdm import tqdm
from typing import Optional
from preprocessing.utils import DatasetFolderStructure
from utils.bounding_box import bbox_from_image_geometry
from config import settings


class QgisProcess(ABC):

    run_name = ''
    params = {}

    def __init__(self, *args, **kwargs) -> None:
        
        for key, val in kwargs:
            self.params[key.upper()] = val


class Reproject(QgisProcess):

    run_name = 'gdal:warpreproject'
    params = {
        'DATA_TYPE': 0, 
        'EXTRA': '', 
        'INPUT': None,
        'MULTITHREADING': False,
        'NODATA': None,
        'OPTIONS': '',
        'OUTPUT': None,
        'RESAMPLING': None,
        'SOURCE_CRS': None,
        'TARGET_CRS': None, 
        'TARGET_EXTENT': None, 
        'TARGET_EXTENT_CRS': None, 
        'TARGET_RESOLUTION': None}

    def __init__(self, input_path: str, output_path: str, target_resolution: float, resampling_method: int = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.params["INPUT"] = input_path
        self.params["OUTPUT"] = output_path
        self.params["TARGET_RESOLUTION"] = target_resolution
        self.params["RESAMPLING"] = resampling_method


class CreateGrid(QgisProcess):
    
    run_name = 'qgis:creategrid'
    params = {
        'TYPE': 2,
        'EXTENT': None,
        'HSPACING': None,
        'VSPACING': None,
        'HOVERLAY': None,
        'VOVERLAY': None,
        'CRS': None,
        'OUTPUT': 'memory:'
    }

    def __init__(self, layer, grid_type: int, spacing: list[float, float], overlay: list[float, float], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.params["TYPE"] = grid_type
        self.params["HSPACING"] = spacing[0]
        self.params["VSPACING"] = spacing[1]
        self.params["HOVERLAY"] = overlay[0]
        self.params["VOVERLAY"] = overlay[1]
        self.params["EXTENT"] = layer.extent()
        self.params["CRS"] = layer.crs()


class Buffer(QgisProcess):

    run_name = "native:buffer"
    params = {
            "INPUT": None,
            "DISTANCE": None,
            "END_CAP_STYLE": 2,
            "DISSOLVE": False,
            "OUTPUT": None,
    }

    def __init__(self, layer, output_path: str, buffer_distance: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.params["INPUT"] = layer
        self.params["OUTPUT"] = output_path
        self.params["DISTANCE"] = buffer_distance


class ClipRasterByMask(QgisProcess):

    run_name = "gdal:cliprasterbymasklayer"
    params = {
        "INPUT": None,
        "MASK": None,
        "NODATA": -9999,
        "ALPHA_BAND": False,
        "CROP_TO_CUTLINE": True,
        "KEEP_RESOLUTION": True,
        "OPTIONS": None,
        "DATA_TYPE": 0,
        "OUTPUT": None
    }

    def __init__(self, layer, mask_layer, output_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.params["INPUT"] = layer
        self.params["MASK"] = mask_layer
        self.params["OUTPUT"] = output_path


def main(
        tif_path: Path, 
        output_path: Path, 
        shp_path: Optional[Path] = None, 
        target_res: Optional[float] = None, 
        buffer_distance: float = 0.25, 
        patch_size: list[int, int] = [640, 640], 
        stride: list[float, float] = [0.5, 0.5], 
        resampling_method: int = 1
    ) -> None:

    # QGIS setup
    from qgis.core import QgsApplication

    os.environ["PYTHON_PATH"] = settings.qgis_plugin_path
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    QgsApplication.setPrefixPath(settings.qgis_path, True)
    qgs = QgsApplication([], False)

    try:
        process(
            tif_path=tif_path,
            output_path=output_path, 
            shp_path=shp_path, 
            target_res=target_res, 
            buffer_distance=buffer_distance, 
            patch_size=patch_size, 
            stride=stride, 
            resampling_method=resampling_method
        )
    finally:
        qgs.exit()


def process(
        tif_path: Path, 
        output_path: Path, 
        shp_path: Optional[Path] = None, 
        target_res: Optional[float] = None, 
        buffer_distance: float = 0.25, 
        patch_size: list[int, int] = [640, 640], 
        stride: list[float, float] = [0.5, 0.5], 
        resampling_method: int = 1
    ) -> None:

    from qgis.core import QgsFeature, QgsProject, QgsRasterLayer, QgsVectorLayer, QgsVectorFileWriter
    # Have to be imported after qgs initialization
    import processing
    from processing.core.Processing import Processing
    Processing.initialize()

    print(f"Tiling tif: {tif_path.name}, shp: {shp_path.name if shp_path else None}")

    # Image parameters
    patch_width, patch_height = patch_size

    # Grid parameters
    grid_type = 2  # Rectangle (Polygon)
    h_stride, v_stride = stride

    # Create subfolders for generated outputs
    folders = DatasetFolderStructure(output_path)

    if target_res is None:
        with Image.open(tif_path) as img:
            
            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
            pixel_size = meta_dict["ModelPixelScaleTag"][:2]
            # Resolution should be equal both directions
            target_res = (pixel_size[0] + pixel_size[1]) / 2

    # Calculate tile size and overlap
    h_spacing = patch_width * target_res
    v_spacing = patch_height * target_res
    h_overlay = h_spacing * h_stride
    v_overlay = v_spacing * v_stride

    map_layer = QgsRasterLayer(str(tif_path), "map")
    QgsProject.instance().addMapLayer(map_layer)
    
    # Get CRS as EPSG code
    map_crs = map_layer.crs()
    crs_string = f"epsg:{map_crs.postgisSrid()}"

    rescaled_tiff_name = tif_path.stem + "_rescaled" + f"{target_res:.2f}".replace(".", "") + "m.tif"
    rescaled_tiff_path = str(output_path / rescaled_tiff_name)

    # Rescale the image
    reproject = Reproject(input_path=map_layer.source(), output_path=rescaled_tiff_path, resampling_method=resampling_method, target_resolution=target_res)
    processing.run(reproject.run_name, reproject.params)

    map_rescaled_layer = QgsRasterLayer(rescaled_tiff_path, rescaled_tiff_name)
    QgsProject.instance().addMapLayer(map_rescaled_layer)

    create_grid = CreateGrid(layer=map_rescaled_layer, grid_type=grid_type, spacing=[h_spacing, v_spacing], overlay=[h_overlay, v_overlay])
    grid_layer = processing.run(create_grid.run_name, create_grid.params)['OUTPUT']
    QgsProject.instance().addMapLayer(grid_layer)

    if shp_path:

        shp_layer = QgsVectorLayer(str(shp_path), "shp", "ogr")
        QgsProject.instance().addMapLayer(shp_layer)
        
        # Create square buffer around each nest
        buffered_shp_path = f"{folders.original_bounding_box_path}/bboxes_{tif_path.stem}.shp"
        buffer = Buffer(layer=shp_layer, output_path=buffered_shp_path, buffer_distance=buffer_distance)
        processing.run(buffer.run_name, buffer.params)

        bboxes = QgsVectorLayer(buffered_shp_path, "bboxes", "ogr")

    total_patches = len(list(grid_layer.getFeatures()))

    for patch in tqdm(grid_layer.getFeatures(), total=total_patches):

        mask = QgsVectorLayer(f"Polygon?crs={crs_string}", "mask", "memory")
        prov = mask.dataProvider()
        prov.addFeatures([patch])
        mask.updateExtents()
        
        patch_name = f"patch{patch.id()}_{tif_path.stem}"
        patch_path = f"{folders.tiled_tif_path}/{patch_name}.tif"

        clip_by_mask = ClipRasterByMask(layer=map_rescaled_layer, mask_layer=mask, output_path=patch_path)
        processing.run(clip_by_mask.run_name, clip_by_mask.params)
        

        with Image.open(patch_path) as img:

            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
            top_left = meta_dict["ModelTiepointTag"][3:5]
            pixel_size = meta_dict["ModelPixelScaleTag"][:2]

            # TODO: Resizing means tags are no longer saved, so this is currently done in create_dataset.py
            # img_width, img_height = img.size

            # if img_width != patch_width or img_height != patch_height:
            #     img = img.resize((patch_width, patch_height), Image.LANCZOS)
            #     img.save(patch_path)
            #     pixel_size = (pixel_size[0] * (img_width / patch_width), pixel_size[1] * (img_height / patch_height))

        if shp_path:

            # save ONLY intersections of bboxes from the current map and patch
            intersections = []
            bbox_cut_to_patches = QgsVectorLayer(f"Polygon?crs={crs_string}", f"bbox_cut_to_patches_{patch_name}", "memory")
            bbox_cut_prov = bbox_cut_to_patches.dataProvider()

            for one_bbox in bboxes.getFeatures():
                if one_bbox.geometry().intersects(patch.geometry()):
                    intersection = one_bbox.geometry().intersection(patch.geometry())
                    intersections.append(intersection.boundingBox())

                    # save geometries to layer to visualize it
                    bbox_cut_feat = QgsFeature()
                    bbox_cut_feat.setGeometry(intersection)
                    bbox_cut_prov.addFeatures([bbox_cut_feat])
                    bbox_cut_to_patches.updateExtents()

            QgsVectorFileWriter.writeAsVectorFormat(
                bbox_cut_to_patches,
                f"{folders.tiled_bounding_box_shp_path}/{patch_name}",
                "utf-8",
                driverName="ESRI Shapefile",
            )

            with open(f"{folders.tiled_bounding_box_txt_path}/{patch_name}.txt", "w") as f:
                for geom in intersections:

                    x_centre, y_centre, width, height = bbox_from_image_geometry(geom, top_left=top_left, pixel_size=pixel_size)

                    # write each item on a new line in YOLO format. Add '0' as nest class
                    f.write(
                        f"0 {x_centre/patch_width} {y_centre/patch_height} {width/patch_width} {height/patch_height}\n"
                    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cut TIFF and ground truth shapefiles into patches')
    parser.add_argument('--tif_path', required=True, type=Path, help='path to the TIFF file')
    parser.add_argument('--shp_path', required=True, type=Path, help='path to the SHP file')
    parser.add_argument('--output_path', required=True, type=Path, help='path to the output directory')
    parser.add_argument('--buffer_distance', type=float, default=0.25, help='buffer distance')
    parser.add_argument('--patch_size', type=int, nargs=2, default=[640, 640], help='patch dimensions')
    parser.add_argument('--target_res', type=float, default=None, help='target resolution')
    parser.add_argument('--resampling_method', type=int, default=1, help='QGIS resampling method (0 = nearest neighbor, 1 = bilinear)')
    parser.add_argument('--stride', type=float, nargs=2, default=[0.5, 0.5], help='patch stride')

    # args = parser.parse_args(
    #     ["--tif_path", "/home/andrew/Antarctic-nests/data/from_bucket/Mapy Turret/Turret_Redu_21_11_23.tif",
    #     "--shp_path", "/home/andrew/Antarctic-nests/data/from_bucket/Shapefiles Turret/21_11_23_Shags_Turret.shp",
    #     "--output_path", "/home/andrew/Antarctic-nests/data/test_output",
    #     "--target_res", "0.02"]
    # )

    args = parser.parse_args()
    main(**vars(args))
