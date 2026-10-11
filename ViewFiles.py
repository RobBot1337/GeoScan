import rasterio
import matplotlib.pyplot as plt

with rasterio.open("satellite_tiles/google_mosaic_google.tif") as g:
    google = g.read()

with rasterio.open("satellite_tiles/esri_mosaic_esri.tif") as e:
    esri = e.read()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(google.transpose(1, 2, 0))
axes[0].set_title("Google")
axes[0].axis("off")

axes[1].imshow(esri.transpose(1, 2, 0))
axes[1].set_title("ESRI")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("all_mosaics.png", dpi=100)
plt.show()