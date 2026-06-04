import os
from PIL import Image


def split_image_with_overlap(input_image_path, output_dir, patch_size=256, overlap=128):
    image = Image.open(input_image_path)
    img_width, img_height = image.size

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]
    step = patch_size - overlap
    patch_count = 0

    for y in range(0, img_height, step):
        if y + patch_size > img_height:
            continue
        for x in range(0, img_width, step):
            if x + patch_size > img_width:
                continue

            box = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(box)

            output_filename = f"{base_name}_{patch_count:04d}.png"
            patch.save(os.path.join(output_dir, output_filename))
            patch_count += 1

    print(f"Done: {input_image_path} -> {patch_count} patches")


def process_all_images_in_folder(input_dir, output_dir):
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif')
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_formats):
            split_image_with_overlap(os.path.join(input_dir, filename), output_dir)


if __name__ == '__main__':
    input_folder = 'path/to/input/images'
    output_folder = 'path/to/output/patches'
    process_all_images_in_folder(input_folder, output_folder)
