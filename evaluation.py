import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sb
from sklearn.metrics import confusion_matrix,accuracy_score, recall_score,f1_score


#---------------Generating label for segmented pre-disaster & post-disaster differences 

img = plt.imread("diffprepost.tif")
img = np.array(img)

#img[img <= 20] = 0     # Noise Cancellation for Ground Truth Image
#img[img >= 21] = 255
#print(img[0])


def slidinggrid(image,stride,filtersize,dii):
    
    file = open("diffprepost.csv", "w")
    file.write("Grid"  + ',' + "label") 
    file.write("\n")
    
    (n_H_prev, n_W_prev) = image.shape
    f = filtersize
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    
    A = np.zeros((n_H, n_W))              
    lp = 0
    for h in range(n_H):                             # loop on the vertical axis of the output volume
        for w in range(n_W): 
                lp=lp+1                              # loop on the horizontal axis of the output volume               
                vert_start = h*stride
                vert_end = h*stride +f
                horiz_start = w*stride
                horiz_end = w*stride + f
                a_prev_slice = image[vert_start:vert_end, horiz_start:horiz_end]
                label= ((a_prev_slice == 255).sum())           # Calculating the numerator for DII
                label = label/ dii                             # Thresholding 
                if (label > 0.01):
                    file.write(str(lp)  + ',' + "1") 
                    file.write("\n")
                else:
                    file.write(str(lp)  + ',' +  "0")
                    file.write("\n")
                    
    print("LP:",lp)   
    file.close()

def DII_denom(image,grid_size):               # Calculating the denominator for DII
    dii = 0
    for i in image:
        for j in i:
            if(j==255):
                dii = dii + 1
            
    return dii/grid_size

img =  img / (255)      #Noise Cancellation for tiff Image ( diff Pre & Post Image)
img = img.astype(int)
print(img[0])


grid_size = 20
No_of_grids = int(1 + (img.shape[0] - grid_size) / grid_size)
dii = DII_denom(img,No_of_grids)
print(No_of_grids, dii)
slidinggrid(img,grid_size,grid_size,dii) 


predicted = pd.read_csv("diffprepost.csv")
ground_truth = pd.read_csv("label.csv")

predicted = predicted.drop('Grid',axis=1)
ground_truth = ground_truth.drop('Grid',axis=1)

#sb.heatmap(confusion_matrix(ground_truth,predicted), annot=True)
print(f1_score(ground_truth,predicted))
