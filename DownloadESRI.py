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
# Совпадает с маленьким范围 из твоего DownloadSputnicImages.py
LAT_MIN, LAT_MAX = 55.75, 55.78
LON_MIN, LON_MAX = 37.55, 37.60

TILE_SIZE = 256
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DELAY = 1.0

# Текущие снимки ESRI (World Imagery)
# Внимание: у ESRI порядок {z}/{y}/{x}, а не как у Google
ESRI_CURRENT_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Шаблон URL исторических снимков Wayback
# release_num — числовой ID, соответствующий снимку на конкретную дату
WAYBACK_URL_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
    "{release_num}/{z}/{y}/{x}"
)

# Список релизов Wayback (вручную подобраны характерные даты)
# Формат: (release_num, "ГГГГ-ММ-ДД", "метка")
# release_num можно получить из @esri/wayback-core или из waybackconfig.json
WAYBACK_RELEASES = [
    (56102, "2023-12-07", "winter_2023"),
    (43072, "2023-01-10", "winter_2023_early"),
    (55136, "2023-07-15", "summer_2023"),
    (46789, "2022-06-30", "summer_2022"),
]


def lat_lon_to_tile(lat, lon, zoom):
    """Переводит широту/долготу в номер тайла по схеме XYZ (Web Mercator)"""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def get_tile_range(lat_min, lat_max, lon_min, lon_max, zoom):
    """Возвращает список всех тайлов (x, y, z), покрывающих заданный прямоугольник"""
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
    """Скачивает один тайл. При ошибке — до 3 попыток с паузой 2 сек"""
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200 and len(resp.content) > 0:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    return False


def download_all_tiles(source_name, url_template, tile_list, extra_format=None):
    """Проходит по списку тайлов и качает только отсутствующие"""
    print(f"\n=== Скачиваю {source_name} ===")
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for x, y, z in tile_list:
        if extra_format:
            url = url_template.format(**extra_format, x=x, y=y, z=z)
        else:
            url = url_template.format(x=x, y=y, z=z)

        save_path = f"{OUTPUT_DIR}/{source_name}/{z}/{x}/{y}.jpg"

        # Если файл уже есть — пропускаем (можно прерывать и продолжать)
        if os.path.exists(save_path):
            skip_count += 1
            continue

        try:
            if download_tile(url, save_path):
                print(f"✓ {source_name} z{z}/x{x}/y{y}")
                ok_count += 1
            else:
                print(f"✗ {source_name} z{z}/x{x}/y{y} (HTTP fail)")
                fail_count += 1
        except Exception as e:
            print(f"✗ {source_name} z{z}/x{x}/y{y} ({e})")
            fail_count += 1

        # Пауза между запросами, чтобы не забанили IP
        time.sleep(DELAY)

    print(f"  Итог {source_name}: +{ok_count} скачано, {skip_count} уже было, {fail_count} ошибок")


def tile_bounds(x, y, z):
    """Переводит номер тайла обратно в метры EPSG:3857 (нужно для GeoTIFF)"""
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
    """Считает transform и размеры итоговой мозаики по списку тайлов"""
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


def stitch_tiles_to_geotiff(source_name, tile_list, output_name=None):
    """Склеивает все тайлы source_name в один GeoTIFF"""
    if output_name is None:
        output_name = f"mosaic_{source_name}.tif"

    # Проверяем, есть ли вообще что склеивать
    existing = [t for t in tile_list if os.path.exists(
        f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{t[0]}/{t[1]}.jpg")]
    if not existing:
        print(f"⚠ {source_name}: нет скачанных тайлов, пропускаю склейку")
        return None

    transform, width, height, xs, ys = get_mosaic_transform(tile_list, ZOOM)
    mosaic = np.zeros((height, width, 3), dtype=np.uint8)

    print(f"\n=== Склеиваю {source_name} → GeoTIFF ===")
    for x, y, _ in tile_list:
        path = f"{OUTPUT_DIR}/{source_name}/{ZOOM}/{x}/{y}.jpg"
        if not os.path.exists(path):
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
            print(f"  Ошибка при чтении тайла {x},{y}: {e}")

    output_path = f"{OUTPUT_DIR}/{output_name}"

    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=np.uint8,
        crs="EPSG:3857",
        transform=transform,
        compress="lzw",
        photometric="RGB",
    ) as dst:
        dst.write(mosaic[:, :, 0], 1)
        dst.write(mosaic[:, :, 1], 2)
        dst.write(mosaic[:, :, 2], 3)

    print(f"✅ Готово: {output_path}")
    return output_path


def pick_season_releases(releases):
    """
    Из списка WAYBACK_RELEASES выбирает по одному релизу на зиму и лето.
    Зима: декабрь, январь, февраль. Лето: июнь, июль, август.
    Берём самые свежие по году.
    Возвращает (зимний_релиз, летний_релиз)
    """
    winter = []
    summer = []

    for release_num, date_str, label in releases:
        year, month, day = map(int, date_str.split("-"))
        item = (release_num, date_str, label, year, month)
        if month in (12, 1, 2):
            winter.append(item)
        elif month in (6, 7, 8):
            summer.append(item)

    # Сортируем по году по убыванию и берём первый
    winter.sort(key=lambda x: x[3], reverse=True)
    summer.sort(key=lambda x: x[3], reverse=True)

    w = winter[0] if winter else None
    s = summer[0] if summer else None
    return w, s


if __name__ == "__main__":
    tile_list = get_tile_range(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ZOOM)
    print(f"Тайлов для ESRI: {len(tile_list)}")
    print(f"Область: lat {LAT_MIN}..{LAT_MAX}, lon {LON_MIN}..{LON_MAX}, zoom {ZOOM}")

    # ---------- 1. Текущие снимки ESRI ----------
    print("\n" + "=" * 50)
    print("ESRI: текущие снимки (World Imagery)")
    print("=" * 50)
    download_all_tiles("esri_current", ESRI_CURRENT_URL, tile_list)
    stitch_tiles_to_geotiff("esri_current", tile_list,
                            output_name="esri_current_mosaic.tif")

    # ---------- 2. Сезонные снимки Wayback ----------
    winter_item, summer_item = pick_season_releases(WAYBACK_RELEASES)

    season_jobs = []
    if winter_item:
        season_jobs.append((f"esri_winter_{winter_item[3]}", winter_item[0], winter_item[1]))
    if summer_item:
        season_jobs.append((f"esri_summer_{summer_item[3]}", summer_item[0], summer_item[1]))

    if not season_jobs:
        print("\n⚠ В списке Wayback не нашлось ни зимы, ни лета")
    else:
        for source_name, release_num, date_str in season_jobs:
            print("\n" + "=" * 50)
            print(f"ESRI Wayback сезон: {source_name} (release {release_num}, {date_str})")
            print("=" * 50)

            fmt = {"release_num": release_num}
            download_all_tiles(source_name, WAYBACK_URL_TEMPLATE, tile_list,
                               extra_format=fmt)
            stitch_tiles_to_geotiff(source_name, tile_list,
                                    output_name=f"{source_name}_mosaic.tif")

    print("\n✅ ESRI: всё готово!")
    print(f"Смотри в папке: {OUTPUT_DIR}")