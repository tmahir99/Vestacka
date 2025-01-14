from PIL import Image
import os

input_image_path = "image.jpg"
image = Image.open(input_image_path)

image_width, image_height = image.size

output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

rows, cols = 3, 3
piece_width = image_width // cols
piece_height = image_height // rows

for row in range(rows):
    for col in range(cols):
        left = col * piece_width
        upper = row * piece_height
        right = left + piece_width
        lower = upper + piece_height
        piece = image.crop((left, upper, right, lower))

        piece_path = os.path.join(output_folder, f"image{row * cols + col + 1}.jpg")
        piece.save(piece_path, quality=95)

print(f"Image pieces saved in the folder: '{output_folder}'")
