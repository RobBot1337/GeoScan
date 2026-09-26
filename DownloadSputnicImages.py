import os
import math
import time
import requests
import numpy as np
import rasterio
from rasterio.transform import from_origin
from PIL import Image
from retry import retry

OUTPUT_DIR = "satellite_tiles"
ZOOM = 14
LAT_MIN, LAT_MAX = 55.55, 55.85
LON_MIN, LON_MAX = 37.35, 37.75

SOURCES = {
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "yandex": "https://sat01.maps.yandex.net/tiles?l=sat&x={x}&y={y}&z={z}"
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DELAY = 1.0
TILE_SIZE = 256


def lat_lon_to_tile(lat, lon, zoom):
    x = (lon + 180) / 360 * (2 ** zoom)
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * (2 ** zoom)
    return int(x), int(y)


def get_tile_range(lat_min, lat_max, lon_min, lon_max, zoom):
    x_min, y_max = lat_lon_to_tile(lat_max, lon_min, zoom)
    x_max, y_min = lat_lon_to_tile(lat_min, lon_max, zoom)

    x_min, x_max = min(x_min, x_max), max(x_min, x_max)
    y_min, y_max = min(y_min, y_max), max(y_min, y_max)

    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y, zoom))
    return tiles


@retry(tries=3, delay=2)
def download_tile(url, save_path):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    if resp.status_code == 200:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        return True
    return False


def download_all_tiles(source_name, url_template, tile_list):
    print(f"\n=== Скачиваю {source_name} ===")
    for x, y, z in tile_list:
        url = url_template.format(x=x, y=y, z=z)
        save_path = f"{OUTPUT_DIR}/{source_name}/{z}/{x}/{y}.jpg"

        if os.path.exists(save_path):
            continue

        try:
            ok = download_tile(url, save_path)
            if ok:
                print(f"✓ {source_name} z{z}/x{x}/y{y}")
            else:
                print(f"✗ {source_name} z{z}/x{x}/y{y} (ошибка)")
        except Exception as e:
            print(f"✗ {source_name} z{z}/x{x}/y{y} ({e})")

        time.sleep(DELAY)


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

    pixel_size_x = (x_right - x_left) / width
    pixel_size_y = (y_top - y_bottom) / height

    transform = from_origin(x_left, y_top, pixel_size_x, pixel_size_y)
    return transform, width, height, xs, ys


def stitch_tiles_to_geotiff(source_name, tile_list, output_name="mosaic.tif"):
    transform, width, height, xs, ys = get_mosaic_transform(tile_list, ZOOM)

    mosaic = np.zeros((height, width, 3), dtype=np.uint8)

    print(f"\n=== Склеиваю {source_name} в GeoTIFF ===")
    for x, y, _ in tile_list:
        path = f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{x}/{y}.jpg"
        if not os.path.exists(path):
            print(f"  Пропускаю {x},{y} (файл не найден)")
            continue

        try:
            img = Image.open(path).convert("RGB")
            if img.size != (TILE_SIZE, TILE_SIZE):
                img = img.resize((TILE_SIZE, TILE_SIZE))
            arr = np.array(img)

            col = xs.index(x)
            row = ys.index(y)
            y0 = row * TILE_SIZE
            x0 = col * TILE_SIZE
            mosaic[y0:y0 + TILE_SIZE, x0:x0 + TILE_SIZE] = arr
        except Exception as e:
            print(f"  Ошибка {x},{y}: {e}")

    output_path = f"{OUTPUT_DIR}/{source_name}_{output_name}"

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=transform,
        compress="lzw",
        photometric="RGB"
    ) as dst:
        dst.write(mosaic[:, :, 0], 1)
        dst.write(mosaic[:, :, 1], 2)
        dst.write(mosaic[:, :, 2], 3)

    print(f"✅ Готово: {output_path}")
    return output_path


if __name__ == "__main__":
    tile_list = get_tile_range(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ZOOM)
    print(f"Всего тайлов: {len(tile_list)}")

    for name, template in SOURCES.items():
        download_all_tiles(name, template, tile_list)

    for name in SOURCES.keys():
        stitch_tiles_to_geotiff(name, tile_list, f"mosaic_{name}.tif")

    print("\n✅ Всё готово! Смотрим в папке:", OUTPUT_DIR)