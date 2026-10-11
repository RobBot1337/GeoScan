import os
import math
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image

from Config import (
    OUTPUT_DIR, ZOOM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    SOURCE_NAMES,
)


TILE_SIZE = 256


def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def get_tile_range(lat_min, lat_max, lon_min, lon_max, zoom):
    x_min, y_max = lat_lon_to_tile(lat_max, lon_min, zoom)
    x_max, y_min = lat_lon_to_tile(lat_min, lon_max, zoom)
    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)
    return [(x, y, zoom) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]


def tile_bounds(x, y, z):
    n = 2.0 ** z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    R = 6378137.0
    x_left = math.radians(lon_left) * R
    x_right = math.radians(lon_right) * R
    y_top = math.log(math.tan(math.pi / 4 + math.radians(lat_top) / 2)) * R
    y_bottom = math.log(math.tan(math.pi / 4 + math.radians(lat_bottom) / 2)) * R
    return x_left, y_bottom, x_right, y_top


def get_mosaic_transform(tile_list, zoom):
    xs = sorted(set(x for x, _, _ in tile_list))
    ys = sorted(set(y for _, y, _ in tile_list))
    x_left, _, _, _ = tile_bounds(xs[0], ys[0], zoom)
    _, _, x_right, _ = tile_bounds(xs[-1], ys[-1], zoom)
    _, _, _, y_top = tile_bounds(xs[0], ys[0], zoom)
    _, y_bottom, _, _ = tile_bounds(xs[-1], ys[-1], zoom)
    width = len(xs) * TILE_SIZE
    height = len(ys) * TILE_SIZE
    px = (x_right - x_left) / width
    py = (y_top - y_bottom) / height
    return from_origin(x_left, y_top, px, py), width, height, xs, ys


def stitch_tiles_to_geotiff(source_name, tile_list, output_name=None):
    if output_name is None:
        output_name = f"mosaic_{source_name}.tif"

    existing = [t for t in tile_list if os.path.exists(
        f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{t[0]}/{t[1]}.jpg")]
    if not existing:
        return None

    transform, width, height, xs, ys = get_mosaic_transform(tile_list, ZOOM)
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)

    for x, y, _ in tile_list:
        path = f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{x}/{y}.jpg"
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB")
        if img.size != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE))
        arr = np.array(img)
        col = xs.index(x)
        row = ys.index(y)
        mosaic[row * TILE_SIZE:(row + 1) * TILE_SIZE,
               col * TILE_SIZE:(col + 1) * TILE_SIZE] = arr

    output_path = f"{OUTPUT_DIR}/{output_name}"
    with rasterio.open(
        output_path, "w",
        driver="GTiff", height=height, width=width, count=3,
        dtype=np.uint8, crs="EPSG:3857", transform=transform,
        compress="lzw", photometric="RGB",
    ) as dst:
        dst.write(mosaic[:, :, 0], 1)
        dst.write(mosaic[:, :, 1], 2)
        dst.write(mosaic[:, :, 2], 3)

    return output_path


if __name__ == "__main__":
    tile_list = get_tile_range(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ZOOM)

    for name in SOURCE_NAMES:
        folder = f"{OUTPUT_DIR}/{name}"
        if os.path.isdir(folder):
            stitch_tiles_to_geotiff(name, tile_list, output_name=f"{name}_mosaic_{name}.tif")