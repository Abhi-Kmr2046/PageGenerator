
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math as m
from scipy.ndimage import rotate
import copy
import xml.etree.ElementTree as ET
import sys
import random
import os
import random
import imutils
import math
#this a new line
#Put the location of strike templates

def show_image(window_name, img):
    
    cv2.imshow(window_name, img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

path = r'C:\Users\abhis\Documents\BTP\line_stroke_300_dpi\train'

def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 

def crop_image_only_outside(img,tol=0):
    mask = img>tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]




tem_files = []
temp_width_list=[]
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            tem_files.append(os.path.join(r, file))
            img=cv2.imread(os.path.join(r,file))
            temp_width_list.append(img.shape[1])


tem_files=sort_list(tem_files,temp_width_list)
temp_w_list_sorted=np.sort(temp_width_list)
temp_w_list_sorted=np.array(temp_w_list_sorted).tolist()



#Location of the Database to be prepared
# ii=1
ei=0
ci=2400
tot=ci
unit_tot=int(tot/6)
print(ei, ci, tot, unit_tot)
def strike_func(img_path, ii):            
        #Location of word images of Database IAM
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        (thresh, img) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        img_hg=38
        
        img = crop_image_only_outside(img, 2)

        h, w = img.shape
        nw = int(w*(img_hg/h))

        if nw > 128:
            img=cv2.resize(img, (128, img_hg))
            color = [255, 255, 255]
            
            top, bottom = (128-img_hg)//2, 128 - img_hg - (128-img_hg)//2
            left, right = 0, 0
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT,value=color)
        else:
            img=cv2.resize(img, (nw, img_hg))

            color = [255, 255, 255]
            
            top, bottom = (128-img_hg)//2, 128 - img_hg - (128-img_hg)//2
            left, right = (128-nw) // 2, 128 - nw - (128-nw) // 2
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT,value=color)



        imgA1= img            
        imgA1 = imgA1.astype(np.float64)
        width=img.shape[1]
                    
        #selecting index of strike template
        if ii==1: # Partial straight
            width_rand=random.randrange(int(0.69*width),int(0.89*width),1)
            minw=min(temp_w_list_sorted, key=lambda x:abs(x-width_rand))
            indx=temp_w_list_sorted.index(minw)
            ren=20 #ren is even number
                
            if indx>=ren/2 and indx<=(len(temp_w_list_sorted)-ren/2):
                    
                indx_add=random.randrange(-ren/2,ren/2,1)
                
            elif indx>(len(temp_w_list_sorted)-ren/2):
                indx_add=random.randrange(-ren,0,1)
                    
            elif indx<ren/2:
                indx_add=random.randrange(0,ren-1,1)
            # Aquire strike template
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)

            (thresh, img_tembw) = cv2.threshold(img_tem, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                            
            angle_ren=int(math.degrees(math.atan(img_hg/img_tem.shape[1])))
            angl=random.randrange(0,5,1)
            angl=pow(-1, angl)*angl
                    #print('ok2') 
            if angl<0:
                angl=360+angl
            img_tembw = imutils.rotate_bound(img_tembw, angl)
                
            img_tembw=cv2.resize(img_tembw,(int((img_tembw.shape[1]*width_rand)/width),int(max(4, (img_tembw.shape[0]*img_tembw.shape[1])/img_tembw.shape[1]) )))
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tem.shape[0]*128)/img_tembw.shape[1])))
            if img_tem.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(124,124))
                    
                    
            imgA = np.zeros((128,128))
            y_offset=66-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img_tembw.shape[0]/4+1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgA[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw                                      
        elif ii==2: # cross
            width_rand=random.randrange(int(0.9*width),int(1.25*width),1)
            minw=min(temp_w_list_sorted, key=lambda x:abs(x-width_rand))
            indx=temp_w_list_sorted.index(minw)
            ren=20 #ren is even number
                
            if indx>=ren/2 and indx<=(len(temp_w_list_sorted)-ren/2):
                    
                indx_add=random.randrange(-ren/2,ren/2,1)
                
            elif indx>(len(temp_w_list_sorted)-ren/2):
                indx_add=random.randrange(-ren,0,1)
                    
            elif indx<ren/2:
                indx_add=random.randrange(0,ren-1,1)
                # Aquire strike template
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
            angle_ren=int(math.degrees(math.atan(24/img_tem.shape[1])))
            angl=random.randrange(5,max(15,angle_ren),1)
                
            img_tem=cv2.resize(img_tem,(int((img.shape[1]*width_rand)/width),int(max(4, (img_tem.shape[0]*img.shape[1])/img_tem.shape[1]) )))
            (thresh, img_tembw) = cv2.threshold(img_tem, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img_tembw = imutils.rotate_bound(img_tembw, angl)

                
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tembw.shape[0]*128)/img_tembw.shape[1])))
            if img_tembw.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(120,120))
                    
                    
            imgA = np.zeros((128,128))
            y_offset=64-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img.shape[0]/4-1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgA[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw

                
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
            angle_ren=int(math.degrees(math.atan(24/img_tem.shape[1])))
            angle_ren=360-angle_ren
            angl=random.randrange(min(345,angle_ren),353,1)
                
            img_tem=cv2.resize(img_tem,(int((img.shape[1]*width_rand)/width),int(max(4, (img_tem.shape[0]*img.shape[1])/img_tem.shape[1]) )))
            (thresh, img_tembw) = cv2.threshold(img_tem, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img_tembw = imutils.rotate_bound(img_tembw, angl)

                
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tembw.shape[0]*128)/img_tembw.shape[1])))
            if img_tembw.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(120,120))
                    
                    
            imgD = np.zeros((128,128))
            y_offset=64-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img.shape[0]/4-1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgD[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw

                
                
            dest_or = cv2.bitwise_or(imgA1, imgA, mask = None) 
            dest_or = cv2.bitwise_or(dest_or, imgD, mask=None)
            imgA = cv2.bitwise_or (imgA, imgD, mask=None)
                    
                    
        elif ii==3: # Multiple
            width_rand=random.randrange(int(0.9*width),int(1.25*width),1)
            minw=min(temp_w_list_sorted, key=lambda x:abs(x-width_rand))
            indx=temp_w_list_sorted.index(minw)
            ren=20 #ren is even number
                
            if indx>=ren/2 and indx<=(len(temp_w_list_sorted)-ren/2):
                    
                indx_add=random.randrange(-ren/2,ren/2,1)
                
            elif indx>(len(temp_w_list_sorted)-ren/2):
                indx_add=random.randrange(-ren,0,1)
                    
            elif indx<ren/2:
                indx_add=random.randrange(0,ren-1,1)
                # Aquire strike template
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
            angle_ren=int(math.degrees(math.atan(24/img_tem.shape[1])))
            angl=random.randrange(7,max(9,angle_ren),1)
                
            img_tem=cv2.resize(img_tem,(int((img.shape[1]*width_rand)/width),int(max(4, (img_tem.shape[0]*img.shape[1])/img_tem.shape[1]) )))
            (thresh, img_tembw) = cv2.threshold(img_tem, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img_tembw = imutils.rotate_bound(img_tembw, angl)

                
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tembw.shape[0]*128)/img_tembw.shape[1])))
            if img_tembw.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(120,120))
                    
                    
            imgA = np.zeros((128,128))
            y_offset=66-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img.shape[0]/4-1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgA[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw

                
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
                                
            img_tem=cv2.resize(img_tem,(int((img.shape[1]*width_rand)/width),int(max(4, (img_tem.shape[0]*img.shape[1])/img_tem.shape[1]) )))
            (thresh, img_tembw) = cv2.threshold(img_tem, 128, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img_tembw = imutils.rotate_bound(img_tembw, angl+random.randrange(-5,5,1))

                
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tembw.shape[0]*128)/img_tembw.shape[1])))
            if img_tembw.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(120,120))
                    
                    
            imgD = np.zeros((128,128))
            y_offset=62-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img.shape[0]/4-1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgD[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw

            imgA = cv2.bitwise_or (imgA, imgD, mask=None)

                
                
        elif ii==4:  # Straight
            width=img.shape[1]
            width_rand=random.randrange(int(0.89*width),int(1.19*width),1)
            minw=min(temp_w_list_sorted, key=lambda x:abs(x-width_rand))
            indx=temp_w_list_sorted.index(minw)
            ren=20 #ren is even number
                
            if indx>=ren/2 and indx<=(len(temp_w_list_sorted)-ren/2):
                    
                indx_add=random.randrange(-ren/2,ren/2,1)
                
            elif indx>(len(temp_w_list_sorted)-ren/2):
                indx_add=random.randrange(-ren,0,1)
                    
            elif indx<ren/2:
                indx_add=random.randrange(0,ren-1,1)
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
            (thresh, img_tembw) = cv2.threshold(img_tem, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
            angle_ren=int(math.degrees(math.atan(img_hg/img_tem.shape[1])))
            angl=random.randrange(0,5,1)
            angl=pow(-1, angl)*angl
                    #print('ok2') 
            if angl<0:
                angl=360+angl
            img_tembw = imutils.rotate_bound(img_tembw, angl)
                
            img_tembw=cv2.resize(img_tembw,(int((img_tembw.shape[1]*width_rand)/width),int(max(4, (img_tembw.shape[0]*img_tembw.shape[1])/img_tembw.shape[1]) )))
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tem.shape[0]*128)/img_tembw.shape[1])))
            if img_tem.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(124,124))
                    
                    
            imgA = np.zeros((128,128))
            y_offset=66-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img_tembw.shape[0]/4+1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgA[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw                                      
            
        elif ii==5:  # Slanted 
            width_rand=random.randrange(int(0.9*width),int(1.1*width),1)
            minw=min(temp_w_list_sorted, key=lambda x:abs(x-width_rand))
            indx=temp_w_list_sorted.index(minw)
            ren=20 #ren is even number
                
            if indx>=ren/2 and indx<=(len(temp_w_list_sorted)-ren/2):
                    
                indx_add=random.randrange(-ren/2,ren/2,1)
                
            elif indx>(len(temp_w_list_sorted)-ren/2):
                indx_add=random.randrange(-ren,0,1)
                    
            elif indx<ren/2:
                indx_add=random.randrange(0,ren-1,1)
                # Aquire strike template
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
            angle_ren=int(math.degrees(math.atan(24/img_tem.shape[1])))
            angl=random.randrange(5,max(15,angle_ren),1)
            angl=pow(-1, angl)*angl
                #print('ok2') 
            if angl<0:
                angl=360+angl

            img_tem=cv2.resize(img_tem,(int((img.shape[1]*width_rand)/width),int(max(4, (img_tem.shape[0]*img.shape[1])/img_tem.shape[1]) )))
            (thresh, img_tembw) = cv2.threshold(img_tem, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img_tembw = imutils.rotate_bound(img_tembw, angl)

                
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tembw.shape[0]*128)/img_tembw.shape[1])))
            if img_tembw.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(120,120))
                    
                    
            imgA = np.zeros((128,128))
            y_offset=64-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img.shape[0]/4-1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgA[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw
            
        elif ii==6: # Partial_Slanted
            width_rand=random.randrange(int(0.7*width),int(0.95*width),1)
            minw=min(temp_w_list_sorted, key=lambda x:abs(x-width_rand))
            indx=temp_w_list_sorted.index(minw)
            ren=20 #ren is even number
                
            if indx>=ren/2 and indx<=(len(temp_w_list_sorted)-ren/2):
                    
                indx_add=random.randrange(-ren/2,ren/2,1)
                
            elif indx>(len(temp_w_list_sorted)-ren/2):
                indx_add=random.randrange(-ren,0,1)
                    
            elif indx<ren/2:
                indx_add=random.randrange(0,ren-1,1)
                # Aquire strike template
            img_tem=cv2.imread(tem_files[indx+indx_add], cv2.IMREAD_GRAYSCALE)
            angle_ren=int(math.degrees(math.atan(img_hg/img_tem.shape[1])))
            angl=random.randrange(5,max(15,angle_ren),1)
            angl=pow(-1, angl)*angl
                #print('ok2') 
            if angl<0:
                angl=360+angl

            img_tem=cv2.resize(img_tem,(int((img.shape[1]*width_rand)/width),int(max(4, (img_tem.shape[0]*img.shape[1])/img_tem.shape[1]) )))
            (thresh, img_tembw) = cv2.threshold(img_tem, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            img_tembw = imutils.rotate_bound(img_tembw, angl)

                
            if img_tembw.shape[1]>128:
                img_tembw=cv2.resize(img_tembw,(128,int((img_tembw.shape[0]*128)/img_tembw.shape[1])))
            if img_tembw.shape[0]>128:
                img_tembw=cv2.resize(img_tembw,(120,120))
                    
                    
            imgA = np.zeros((128,128))
            y_offset=64-int(img_tembw.shape[0]/2)+ random.randrange(0,int(img.shape[0]/4-1))
            x_offset=64-int(img_tembw.shape[1]/2)
            imgA[y_offset:y_offset+img_tembw.shape[0], x_offset:x_offset+img_tembw.shape[1]] = img_tembw




            
        if ii==0:
            return imgA1
        
        imgA1 = 255-imgA1
        result = cv2.addWeighted(imgA, 0.5, imgA1, 0.5, 0)

        result = result.astype(np.uint8)
        result = cv2.bitwise_not(result)
        (thresh, result) = cv2.threshold(result, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        
        return result


 
