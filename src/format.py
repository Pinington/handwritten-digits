import numpy as np
from skimage.measure import label, regionprops

def divide_image(image):
    ''' Divides an image into multiple images containing every connex composant '''
    connected_components = []
    labeled = label(image > 0)

    for region in regionprops(labeled):
        component = (labeled == region.label).astype(image.dtype)
        component = component[region.slice]
        min_col = region.bbox[1]  # x_min
        connected_components.append((min_col, component))

    connected_components.sort(key=lambda x: x[0])  # sort left → right
    return [c[1] for c in connected_components]


def pad_to_square(img, margin_ratio=0.2):
    h, w = img.shape
    base_size = max(h, w)

    margin = int(base_size * margin_ratio)
    size = base_size + 2 * margin

    padded = np.zeros((size, size), dtype=img.dtype)

    y_offset = margin + (base_size - h) // 2
    x_offset = margin + (base_size - w) // 2

    padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

    return padded