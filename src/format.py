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
    target_size = max(h, w)  # The square size is determined by the larger dimension

    # Calculate horizontal padding
    total_padding = target_size - w
    left_padding = total_padding // 2 + int(target_size * margin_ratio)
    right_padding = total_padding - total_padding // 2 + int(target_size * margin_ratio)

    # Create padded array
    padded = np.zeros((h, w + left_padding + right_padding), dtype=img.dtype)
    
    # Place original image
    padded[:, left_padding:left_padding+w] = img

    return padded