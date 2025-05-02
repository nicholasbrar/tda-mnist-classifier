import numpy as np
import cv2

def binarize_digit(flat_img, threshold = 0.3, morph_clean=True):
    img_2d = flat_img.reshape(28, 28)
    binary_img = (img_2d > (threshold * 255)).astype(np.uint8) * 255
    
    # Apply morphological opening to remove small noise if morph_clean is True
    if morph_clean:

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    return binary_img