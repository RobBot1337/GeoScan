import rasterio
import matplotlib.pyplot as plt

with rasterio.open("satellite_tiles/combined.tif") as b:
    block = b.read()

with rasterio.open("satellite_tiles/combined_pixelwise.tif") as p:
    pixel = p.read()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

axes[0].imshow(block.transpose(1, 2, 0))
axes[0].set_title("Блочное склеивание")
axes[0].axis("off")

axes[1].imshow(pixel.transpose(1, 2, 0))
axes[1].set_title("Попиксельное склеивание")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("all_mosaics.png", dpi=100)
plt.show()