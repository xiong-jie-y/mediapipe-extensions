import numpy as np
import cv2

def overlay_rect_with_opacity(overlaid_image, rect):
    x, y, w, h = rect
    sub_img = overlaid_image[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    # Putting the image back to its position
    overlaid_image[y:y+h, x:x+w] = res