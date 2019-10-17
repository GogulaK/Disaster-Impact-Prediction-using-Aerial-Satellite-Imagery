import tifffile as tiff
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
#Image has channel infront
#result_ for pre result
#result__ for post result
dict = {'crops':0,'water':1,'buildings':2,'roads':3,'trees':4}

def splitResult(name,image):
    for i in range(5):
        tiff.imsave(name+'_'+str(i)+'.tif',image[:,:,i])

def readsplit(*args):
    if len(args) > 5:
        print("Args should be less than or equal to 5:")
        return
    ch = [dict[x] for x in args]
    tid='result'
    grayImage = tiff.imread('result_1.tif')

    img1 = tiff.imread('result.tif')
    print(img1.shape)
    id='postresult'
    img2 = tiff.imread('postresult.tif')
    
    img1 = img1.transpose([1, 2, 0])
    print(img1.shape)
    
    img2 = img2.transpose([1, 2, 0])
    print(img2.shape)
    
    preinput=np.zeros(shape=(grayImage.shape[0],grayImage.shape[1],len(ch)))
    postinput=np.zeros(shape=(grayImage.shape[0],grayImage.shape[1],len(ch)))
    
    print("preinp",preinput.shape)
    print("postinp",preinput.shape)
    
    for i in range(len(ch)):
        print("CH:",ch[i])
        preinput[:,:,i] = img1[:,:,ch[i]]    
        postinput[:,:,i] = img2[:,:,ch[i]]

    summedPreInput = preinput.sum(axis=2)
    
    summedPostInput = postinput.sum(axis=2)
    tiff.imsave('summedpre.tif',255*summedPreInput[:,:].astype('uint16'))
    tiff.imsave('summedpost.tif',255*summedPostInput[:,:].astype('uint16'))
    

    dife =  summedPreInput - summedPostInput 
    numzeros=(dife < 0).sum()
    print("count",numzeros)
    print(dife.shape)
    dife[dife < 10] = 0
    dife[dife >= 10] = 255
    tiff.imsave('preinput.tif',255*(preinput[:,:,:]).astype('uint16'))
    tiff.imsave('postinput.tif',(255*postinput[:,:,:]).astype('uint16'))
    tiff.imsave('diffprepost.tif',(255*dife[:,:]).astype('uint16'))

    
test_id='result'
img = tiff.imread('{}.tif'.format(test_id))
img_m = img.transpose([1, 2, 0])
splitResult(test_id,img_m)
img = tiff.imread('{}.tif'.format(test_id))
img_m = img.transpose([1, 2, 0])
splitResult(test_id,img_m)
#Preference which is needed 
p = ['buildings','roads','trees']

readsplit(*p)
