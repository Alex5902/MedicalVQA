# create_dummy_image.py
from PIL import Image
import os

# --- Configuration ---
output_dir = "/home/alex.ia/OmniMed/OmniMedVQA/OmniMedVQA/Images/DUMMY" 
filename = "dummy_black_336.png" 
image_size = (336, 336)
color = "black"
# -------------------

os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, filename)

img = Image.new('RGB', image_size, color=color)
img.save(output_path)

print(f"Dummy image saved to: {output_path}")