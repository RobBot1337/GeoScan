import os
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt

from Config import OUTPUT_DIR

BRIGHT_THR = 170
SAT_THR = 35
LOCAL_CONTRAST_THR = 12.0
FEATHER_SIGMA = 4.0
USE_CLOUD_MASK = True

NAME_A = "google"
NAME_B = "esri"
OUTPUT_NAME = "combined_pixelwise.tif"
OUTPUT_PREVIEW = "combined_pixelwise.png"
OUTPUT_COMPARE = "combined_compare.png"
BLOCK_REFERENCE = "combined_block.tif"


def _cloud_mask_pixelwise(arr_rgb, bright_thr=BRIGHT_THR, sat_thr=SAT_THR,
                          local_contrast_thr=LOCAL_CONTRAST_THR):
    a = arr_rgb.astype(np.float32)
    mean = a.mean(axis=0)
    sat = a.max(axis=0) - a.min(axis=0)
    blur = cv2.GaussianBlur(mean, (0, 0), sigmaX=8.0)
    contrast = np.abs(mean - blur)

    m = ((mean > bright_thr).astype(np.float32)
         * (sat < sat_thr).astype(np.float32)
         * (contrast < local_contrast_thr).astype(np.float32))

    m = cv2.GaussianBlur(m, (0, 0), sigmaX=3.0)
    m_u8 = (m > 0.4).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, k, iterations=2)
    m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, k, iterations=1)
    return m_u8.astype(np.float32) / 255.0


def mosaic_path(name):
    return f"{OUTPUT_DIR}/{name}_mosaic_{name}.tif"


def load_mosaic(name):
    path = mosaic_path(name)
    if not os.path.exists(path):
        return None, None
    with rasterio.open(path) as src:
        return src.read(), src.profile


def save_geotiff(arr, profile, output_name):
    out = f"{OUTPUT_DIR}/{output_name}"
    profile = profile.copy()
    profile.update(dtype=rasterio.uint8, count=3, compress="lzw", photometric="RGB")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(arr)
    return out


def merge_two_sources_pixelwise(name_a=NAME_A, name_b=NAME_B,
                                output_name=OUTPUT_NAME,
                                bright_thr=BRIGHT_THR,
                                sat_thr=SAT_THR,
                                local_contrast_thr=LOCAL_CONTRAST_THR,
                                feather_sigma=FEATHER_SIGMA,
                                use_cloud_mask=USE_CLOUD_MASK):
    img_a, profile = load_mosaic(name_a)
    img_b, _ = load_mosaic(name_b)

    if img_a is None or img_b is None:
        raise FileNotFoundError(
            f"Одна из мозаик не найдена:\n  {mosaic_path(name_a)}\n  {mosaic_path(name_b)}"
        )

    if img_a.shape != img_b.shape:
        raise ValueError(f"Размеры мозаик не совпадают: {img_a.shape} vs {img_b.shape}")

    a_f = img_a.astype(np.float32)
    b_f = img_b.astype(np.float32)

    if use_cloud_mask:
        cloud_a = _cloud_mask_pixelwise(a_f, bright_thr, sat_thr, local_contrast_thr)
        cloud_b = _cloud_mask_pixelwise(b_f, bright_thr, sat_thr, local_contrast_thr)
    else:
        H, W = img_a.shape[1:]
        cloud_a = np.zeros((H, W), dtype=np.float32)
        cloud_b = np.zeros((H, W), dtype=np.float32)

    clean_a = 1.0 - cloud_a
    clean_b = 1.0 - cloud_b

    denom = clean_a + clean_b
    denom[denom < 1e-6] = 1e-6

    w_a = clean_a / denom
    w_b = 1.0 - w_a

    both_cloudy = (cloud_a > 0.5) & (cloud_b > 0.5)
    w_a[both_cloudy] = 0.5
    w_b[both_cloudy] = 0.5

    w_a = cv2.GaussianBlur(w_a, (0, 0), sigmaX=feather_sigma)
    w_b = cv2.GaussianBlur(w_b, (0, 0), sigmaX=feather_sigma)

    s = w_a + w_b
    s[s < 1e-6] = 1.0
    w_a /= s
    w_b /= s

    w_a3 = w_a[None, :, :]
    w_b3 = w_b[None, :, :]

    result = np.clip(w_a3 * a_f + w_b3 * b_f, 0, 255).astype(np.uint8)

    save_geotiff(result, profile, output_name)

    return result, profile


if __name__ == "__main__":
    result_pixel, _ = merge_two_sources_pixelwise(
        name_a=NAME_A,
        name_b=NAME_B,
        output_name=OUTPUT_NAME,
        bright_thr=BRIGHT_THR,
        sat_thr=SAT_THR,
        local_contrast_thr=LOCAL_CONTRAST_THR,
        feather_sigma=FEATHER_SIGMA,
        use_cloud_mask=USE_CLOUD_MASK,
    )

    block_path = f"{OUTPUT_DIR}/{BLOCK_REFERENCE}"
    if os.path.exists(block_path):
        with rasterio.open(block_path) as src:
            result_block = src.read()

        if result_block.shape == result_pixel.shape:
            diff = np.abs(result_block.astype(np.int16) - result_pixel.astype(np.int16)).astype(np.uint8)
            diff_vis = np.clip(diff.astype(np.int16) * 3, 0, 255).astype(np.uint8)

            fig, ax = plt.subplots(1, 3, figsize=(22, 8))
            ax[0].imshow(result_block.transpose(1, 2, 0))
            ax[0].set_title("Block 512px")
            ax[0].axis("off")
            ax[1].imshow(result_pixel.transpose(1, 2, 0))
            ax[1].set_title("Результат")
            ax[1].axis("off")
            ax[2].imshow(diff_vis.transpose(1, 2, 0))
            ax[2].set_title("|diff| ×3")
            ax[2].axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_COMPARE, dpi=110)
            plt.show()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(result_pixel.transpose(1, 2, 0))
            ax.set_title("Результат")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(OUTPUT_PREVIEW, dpi=110)
            plt.show()
    else:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(result_pixel.transpose(1, 2, 0))
        ax.set_title("Результат")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUTPUT_PREVIEW, dpi=110)
        plt.show()