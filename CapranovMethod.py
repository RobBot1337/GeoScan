import os
import math
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
from rasterio.transform import from_origin
from PIL import Image
from Config import OUTPUT_DIR, ZOOM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, SOURCE_NAMES

TILE_SIZE = 256
CLOUD_THRESHOLD = 160
BLUR_SIGMA = 15  # Сила размытия для плавных переходов


def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    return (int((lon + 180) / 360 * n),
            int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n))


def get_tile_range(lat_min, lat_max, lon_min, lon_max, zoom):
    x1, y1 = lat_lon_to_tile(lat_max, lon_min, zoom)
    x2, y2 = lat_lon_to_tile(lat_min, lon_max, zoom)
    return [(x, y, zoom) for x in range(min(x1, x2), max(x1, x2) + 1) for y in range(min(y1, y2), max(y1, y2) + 1)]


def tile_bounds(x, y, z):
    n, R = 2.0 ** z, 6378137.0
    lon_l, lon_r = x / n * 360.0 - 180.0, (x + 1) / n * 360.0 - 180.0
    lat_t = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_b = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (math.radians(lon_l) * R, math.log(math.tan(math.pi / 4 + math.radians(lat_b) / 2)) * R,
            math.radians(lon_r) * R, math.log(math.tan(math.pi / 4 + math.radians(lat_t) / 2)) * R)


def get_mosaic_transform(tile_list, zoom):
    xs, ys = sorted(set(x for x, _, _ in tile_list)), sorted(set(y for _, y, _ in tile_list))
    x_l, _, _, _ = tile_bounds(xs[0], ys[0], zoom)
    _, _, x_r, _ = tile_bounds(xs[-1], ys[-1], zoom)
    _, _, _, y_t = tile_bounds(xs[0], ys[0], zoom)
    _, y_b, _, _ = tile_bounds(xs[-1], ys[-1], zoom)
    w, h = len(xs) * TILE_SIZE, len(ys) * TILE_SIZE
    return from_origin(x_l, y_t, (x_r - x_l) / w, (y_t - y_b) / h), w, h, xs, ys


def stitch_tiles_to_geotiff(source_name, tile_list, output_name=None):
    output_name = output_name or f"mosaic_{source_name}.tif"
    if not any(os.path.exists(f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{t[0]}/{t[1]}.jpg") for t in tile_list):
        return None
    transform, width, height, xs, ys = get_mosaic_transform(tile_list, ZOOM)
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)
    for x, y, _ in tile_list:
        path = f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{x}/{y}.jpg"
        if os.path.exists(path):
            img = Image.open(path).convert("RGB").resize((TILE_SIZE, TILE_SIZE))
            mosaic[ys.index(y) * TILE_SIZE:(ys.index(y) + 1) * TILE_SIZE, xs.index(x) * TILE_SIZE:(xs.index(
                x) + 1) * TILE_SIZE] = np.array(img)
    output_path = f"{OUTPUT_DIR}/{output_name}"
    with rasterio.open(output_path, "w", driver="GTiff", height=height, width=width, count=3,
                       dtype=np.uint8, crs="EPSG:3857", transform=transform, compress="lzw", photometric="RGB") as dst:
        for i in range(3): dst.write(mosaic[:, :, i], i + 1)
    return output_path


def blend_mosaics_smoothly(source_names, tile_list, output_name="composite_smooth.tif"):
    mosaics = {}
    for name in source_names:
        path = f"{OUTPUT_DIR}/{name}_mosaic_{name}.tif"
        if os.path.exists(path):
            with rasterio.open(path) as src: mosaics[name] = src.read()

    if len(mosaics) < 2:
        print("Нужно минимум 2 источника для смешивания.")
        return None

    names = list(mosaics.keys())
    base_img = mosaics[names[0]].astype(np.float32)
    H, W = base_img.shape[1:]

    # Итоговое изображение и общий вес (сумма уверенности)
    result_img = np.zeros_like(base_img)
    total_weight = np.zeros((H, W), dtype=np.float32)

    for name in names:
        img = mosaics[name].astype(np.float32)

        # Вычисляем "облачность" (чем ярче - тем больше вес облака)
        gray = img.mean(axis=0)
        cloud_mask = np.clip((gray - CLOUD_THRESHOLD) / (255.0 - CLOUD_THRESHOLD), 0, 1)

        # ВАЖНО: Защита от "черных дыр" (отсутствия данных)
        # Если пиксель очень темный (нет данных), мы даем ему нулевой вес.
        valid_mask = (gray > 5).astype(np.float32)

        # Вес пикселя = (1 - облачность) * валидность
        weight = (1.0 - cloud_mask) * valid_mask

        # Слегка размываем веса, чтобы швы не были резкими
        weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=BLUR_SIGMA)

        # Добавляем к общему результату
        for c in range(3):
            result_img[c] += img[c] * weight
        total_weight += weight

    # Нормализация (чтобы не было пересветов)
    total_weight = np.maximum(total_weight, 1e-5)  # Защита от деления на 0
    for c in range(3):
        result_img[c] /= total_weight

    result = np.clip(result_img, 0, 255).astype(np.uint8)

    transform, width, height, _, _ = get_mosaic_transform(tile_list, ZOOM)
    output_path = f"{OUTPUT_DIR}/{output_name}"
    with rasterio.open(output_path, "w", driver="GTiff", height=height, width=width, count=3,
                       dtype=np.uint8, crs="EPSG:3857", transform=transform, compress="lzw") as dst:
        dst.write(result)
    return result


if __name__ == "__main__":
    tile_list = get_tile_range(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ZOOM)
    for name in SOURCE_NAMES:
        if os.path.isdir(f"{OUTPUT_DIR}/{name}"):
            stitch_tiles_to_geotiff(name, tile_list, output_name=f"{name}_mosaic_{name}.tif")

    # Используем новую функцию плавного смешивания
    result = blend_mosaics_smoothly(SOURCE_NAMES, tile_list, output_name="composite_smooth.tif")

    if result is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(result.transpose(1, 2, 0))
        plt.title("Плавное смешивание на основе облачности")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("composite_smooth.png", dpi=100)
        plt.show()