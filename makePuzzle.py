from PIL import Image, ImageDraw, ImageFont
import random
import os

# Load the image
input_image_path = "image.jpg"  # Replace with the path to your image
image = Image.open(input_image_path)

# Resize the image to ensure it's square and suitable for a 3x3 grid
image = image.resize((300, 300))  # Adjust size as needed
image_width, image_height = image.size

# Create a folder for output if not exists
output_folder = "images"
os.makedirs(output_folder, exist_ok=True)

# Define grid dimensions
rows, cols = 3, 3
piece_width = image_width // cols
piece_height = image_height // rows

# Split the image into 9 pieces
for row in range(rows):
    for col in range(cols):
        left = col * piece_width
        upper = row * piece_height
        right = left + piece_width
        lower = upper + piece_height
        piece = image.crop((left, upper, right, lower))

        # Save the piece with a specific name
        piece_path = os.path.join(output_folder, f"image{row * cols + col + 1}.jpg")
        piece.save(piece_path)

print(f"Image pieces saved in the folder: '{output_folder}'")