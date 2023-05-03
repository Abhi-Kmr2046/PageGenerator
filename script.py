from Levenshtein import distance
import argparse
import cv2, os
import numpy as np
from strike import strike_func




mapping_file = "mapping.csv"

def similar_words(s):
    min_dis = 1000000
    # print(lis, s)
    lis = []
    with open(mapping_file, 'r') as f:
        for line in f:
            (word, path) = line.split('\t')
            min_dis = min(min_dis, distance(s, word))
    with open(mapping_file, 'r') as f:
        for line in f:
            (word, path) = line.split('\t')
            path = path.rstrip()
            if distance(s, word) == min_dis:
                lis.append((word, path))    
    print("min_dis: ", min_dis)
    return lis

def path_of_images(words):
    lis = []
    for item in words:
        print(item[1])
        if not os.path.exists(item[1]):
            print(f'{item[1]} does not exist')
            break
        # img = cv2.imread(item[1])
        lis.append(item[1])
    return lis

def show_image(window_name, img):
    
    cv2.imshow(window_name, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_tupule(image):
    # Window name in which image is displayed
    window_name = image[0]
    print(window_name)
    show_image(window_name, image[1])

# path = "./words/a01/a01-000u/a01-000u-00-00.png"
# if not os.path.exists(path):
#     print(f'does not exist')
# exit()

def crop_image_only_outside(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def binarize_image(gray_image, thresh = None):

    img_binary = 0
    if thresh is None:
        (thresh, img_binary) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        img_binary = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]

    return (thresh, img_binary)



def image_resize_pad(image, desired_size = 256):
    old_size = image.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    (thresh, img_binary) = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img_binary

def fill_color(image, color):
    image[np.where((image==[255,255,255]).all(axis=2))] = color
    return image

def fill_color_slider(image) :
    
    windowName ="Open CV Color Palette"  
    # window name
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow(windowName) 
    # create trackbars for color change
    cv2.createTrackbar('R',windowName,0,255,lambda args: None)
    cv2.createTrackbar('G',windowName,0,255,lambda args: None)
    cv2.createTrackbar('B',windowName,0,255,lambda args: None)
    r, g, b = 255,255,255
    while(1):
        cv2.imshow(windowName, img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R',windowName)
        g = cv2.getTrackbarPos('G',windowName)
        b = cv2.getTrackbarPos('B',windowName)

        img[:] = [b,g,r]
           
    cv2.destroyAllWindows()

    print("RGB ", r, g, b)
    # return 0
    return fill_color(image,[b,g,r])

def crop_image_only_outside(img,tol=0):
    mask = img>tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def crop_final_image(img):
    height, width = img.shape
    left = 0
    top = height // 4
    right = width
    bottom = height - height // 4
    img_cropped = img[top:bottom, left:right]
    return img_cropped

def generate_word(word, strike, color):
    # path = "./words/a01/a01-000x/a01-000x-00-06.png"

    input_word = word

    words = similar_words(input_word)

    word = words[0]

    striked = strike_func(word[1], strike)

    print("striked1 ", striked.shape)
    
    # show_image("wind", striked)


    striked = crop_final_image(striked)

    print("striked2 ", striked.shape)
    # show_image("wind", striked)

    bgr = cv2.merge((striked, striked, striked))
    print("bgr ", bgr.shape)


    filled = fill_color(bgr, color)
    # show_image("filled", filled)
    return (bgr, filled)

def main_test(args):
    # path = "./words/a01/a01-000x/a01-000x-00-06.png"

    input_word = args.word

    words = similar_words(input_word)

    print(words)

    word = words[0]

    print("word is: ", word[0])
    striked = strike_func(word[1], args.strike)
    show_image("window", striked)
    print("striked ", striked.shape)
    

    bgr = cv2.merge((striked, striked, striked))
    print("bgr ", bgr.shape)
    show_image("filled", bgr)

    filled = fill_color_slider(bgr)
    show_image("filled", filled)
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-w","--word", help="Input word", type=str, required=True)
    parser.add_argument("-s","--strike", help="Path of strike", type=int, required=False)

    args = parser.parse_args()

    # main(args)
    # main_test(args)
    main_test(args)

    