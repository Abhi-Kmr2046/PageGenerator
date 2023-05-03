import cv2
import numpy as np
from script import generate_word, show_image
PAGE_SIZE = (960, 675, 3)

# img = generate_word('a', 2, [100,100,100])
# cv2.imwrite("img/word.jpg", img)

# show_image("image", img)

PAGE_GT = np.ones(PAGE_SIZE)*255
PAGE_GEN = np.ones(PAGE_SIZE)*255

def replace_pixel_GT(big_image, small_image, position):
    # Get dimensions of small image
    height, width, _ = small_image.shape

    # Get dimensions of big image
    big_height, big_width, _ = big_image.shape

    # Replace pixel
    big_image[position[0]:min(position[0]+height, big_height), position[1]:min(position[1]+width, big_width)] = small_image

    return big_image


def add_word(word:str, pos:tuple, ii:int):
    bgr, filled = generate_word(word, ii, [255,255,255])
    replace_pixel_GT(PAGE_GEN, bgr, pos)
    replace_pixel_GT(PAGE_GT, filled, pos)
    return

add_word("hello", 2, (0,0))



show_image("1", PAGE_GT)
show_image("2", PAGE_GEN)
cv2.imwrite("img/GT.jpg", PAGE_GT)
cv2.imwrite("img/GEN.jpg", PAGE_GEN)

