import rasterio
import matplotlib.pyplot as plt

with rasterio.open("satellite_tiles/google_mosaic_google.tif") as g:
    google = g.read()
    print("Google:", g.crs, g.bounds, g.shape)

with rasterio.open("satellite_tiles/yandex_mosaic_yandex.tif") as y:
    yandex = y.read()
    print("Yandex:", y.crs, y.bounds, y.shape)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(google.transpose(1, 2, 0))  # (3,H,W) -> (H,W,3)
axes[0].set_title("Google")
axes[0].axis("off")

axes[1].imshow(yandex.transpose(1, 2, 0))
axes[1].set_title("Yandex")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("both_mosaics.png", dpi=100)
plt.show()