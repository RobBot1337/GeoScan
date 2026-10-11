OUTPUT_DIR = "satellite_tiles"
ZOOM = 14

LAT_MIN, LAT_MAX = 55.74, 55.84
LON_MIN, LON_MAX = 49.05, 49.20

SEASON = "summer"

SOURCE_NAMES = ["google", "esri", "esri_season"]

SEASON_MONTHS = {
    "winter": (12, 1, 2),
    "spring": (3, 4, 5),
    "summer": (6, 7, 8),
    "autumn": (9, 10, 11),
}