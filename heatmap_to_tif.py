import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
s = '/Users/mahendrensundararajan/Desktop/CodeDLSubmission/project/'


def process(foldername):
    os.chdir(s+foldername)
    im1=plt.imread('summedpre.tif')
    im2=plt.imread('summedpost.tif')
    im1 = np.array(im1)
    im2 = np.array(im2)
    dif=np.subtract(im1,im2)
    numzeros=(dif < 0).sum()
    print(numzeros)
    dif[dif <= 2000] = 65535
    plt.imshow(dif,cmap='hot',interpolation='nearest')
    plt.savefig('heatmap_nothrehold.tif')

if __name__ == "__main__":
    #python heat_map_to_tif.py foldername
    process(sys.argv[1])
    