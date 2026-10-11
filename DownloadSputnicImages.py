import os
import math
import time
import requests
from retry import retry
from datetime import datetime

from Config import (
    OUTPUT_DIR, ZOOM, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
    SEASON, SEASON_MONTHS,
)


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
DELAY = 1.0

SOURCES = {
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}

WAYBACK_RELEASES = [
    (56102, "2023-12-07"),
    (55136, "2023-07-15"),
    (43072, "2023-01-10"),
    (46789, "2022-06-30"),
    (39284, "2022-02-20"),
    (31876, "2021-06-10"),
    (26376, "2020-12-20"),
    (21521, "2020-06-15"),
    (20337, "2024-09-19"),
    (56450, "2024-10-10"),
    (49849, "2024-11-18"),
    (52930, "2024-05-09"),
    (12428, "2024-06-06"),
    (60013, "2024-03-07"),
]

WAYBACK_TILE_TEMPLATE = (
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
    "World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/"
    "{release_num}/{z}/{y}/{x}"
)


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

    tiles = []
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            tiles.append((x, y, zoom))
    return tiles


@retry(tries=3, delay=2)
def download_tile(url, save_path):
    resp = requests.get(url, headers=HEADERS, timeout=15)
    if resp.status_code == 200 and len(resp.content) > 100:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(resp.content)
        return True
    return False


def download_all_tiles(source_name, url_template, tile_list, extra_format=None):
    for x, y, z in tile_list:
        if extra_format:
            url = url_template.format(**extra_format, x=x, y=y, z=z)
        else:
            url = url_template.format(x=x, y=y, z=z)

        save_path = f"{OUTPUT_DIR}/{source_name}/{z}/{x}/{y}.jpg"

        if os.path.exists(save_path):
            continue

        try:
            download_tile(url, save_path)
        except Exception:
            pass

        time.sleep(DELAY)


def pick_season_release(season):
    months = SEASON_MONTHS[season.lower()]
    candidates = []

    for release_num, date_str in WAYBACK_RELEASES:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        if dt.month in months:
            candidates.append((release_num, date_str, dt))

    if not candidates:
        raise RuntimeError(f"Не нашлось релизов для сезона '{season}'")

    candidates.sort(key=lambda x: x[2], reverse=True)
    best = candidates[0]
    return best[0], best[1]


if __name__ == "__main__":
    tile_list = get_tile_range(LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, ZOOM)

    for source_name, template in SOURCES.items():
        download_all_tiles(source_name, template, tile_list)

    release_num, release_date = pick_season_release(SEASON)

    wayback_name = "esri_season"
    fmt = {"release_num": release_num}

    download_all_tiles(wayback_name, WAYBACK_TILE_TEMPLATE, tile_list, extra_format=fmt)