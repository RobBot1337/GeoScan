import os
import numpy as np
from PIL import Image
from Config import OUTPUT_DIR, ZOOM


def has_black_edges(tile_img, threshold=10):
    arr = np.array(tile_img)
    
    top_edge = arr[:5, :, :]
    bottom_edge = arr[-5:, :, :]
    left_edge = arr[:, :5, :]
    right_edge = arr[:, -5:, :]
    
    def count_dark(edge_part):
        return np.sum(np.all(edge_part < threshold, axis=2))
    
    total_dark = count_dark(top_edge) + count_dark(bottom_edge) + count_dark(left_edge) + count_dark(right_edge)
    
    return total_dark > 100


def fill_black_edges(base_tile, donor_tile, threshold=10):
    base_arr = np.array(base_tile)
    donor_arr = np.array(donor_tile)
    
    black_mask = np.all(base_arr < threshold, axis=2)
    base_arr[black_mask] = donor_arr[black_mask]
    
    return Image.fromarray(base_arr)


def main():
    
    source_name = "esri"
    donor_name = "google"
    
    source_dir = f"{OUTPUT_DIR}/{source_name}/{ZOOM}"
    donor_dir = f"{OUTPUT_DIR}/{donor_name}/{ZOOM}"
    
    if not os.path.exists(source_dir):
        print(f"Ошибка: нет папки {source_dir}.")
        return

    fixed_count = 0

    for x in os.listdir(source_dir):
        x_path = os.path.join(source_dir, x)
        if not os.path.isdir(x_path): 
            continue
        
        for y_file in os.listdir(x_path):
            if not y_file.endswith(".jpg"): 
                continue
            
            source_file = os.path.join(x_path, y_file)
            donor_file = os.path.join(donor_dir, x, y_file)
            
            if not os.path.exists(donor_file):
                continue

            img_source = Image.open(source_file).convert("RGB")
            img_donor = Image.open(donor_file).convert("RGB")
            
            if has_black_edges(img_source):
                fixed_img = fill_black_edges(img_source, img_donor)
                fixed_img.save(source_file)
                fixed_count += 1
                print(f"Исправлен тайл: {x}/{y_file}")

    print(f"\n Исправлено тайлов: {fixed_count}")


if __name__ == "__main__":
    main()