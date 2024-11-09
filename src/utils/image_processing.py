from PIL import Image
import numpy as np

def preprocess_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    max_size = 1000
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image
