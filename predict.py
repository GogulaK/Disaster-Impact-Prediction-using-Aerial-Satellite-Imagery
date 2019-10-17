import math
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from train_unet import weights_path, get_model, normalize, dimpatchsz, outputClasses


def predictoutput(testdata, model, dimpatch=160, numclass=5):
    h = testdata.shape[0]
    w = testdata.shape[1]
    c = testdata.shape[2]
    print("IMG HEIGHT:",h)
    print("IMG WIDTH:",w)
    print("IMG channels:",c)
    verticalpatches = math.ceil(h /dimpatch)
    horizontalpatches = math.ceil(w/dimpatch)
    addedh = dimpatch * verticalpatches
    addedw = dimpatch * horizontalpatches
    extended = np.zeros(shape=(addedh, addedw, c), dtype=np.float32)
    extended[:h, :w, :] = testdata
    for i in range(h, addedh):
        extended[i, :, :] = extended[2 * h - i - 1, :, :]
    for j in range(w, addedw):
        extended[:, j, :] = extended[:, 2 * w - j - 1, :]

    patches_list = []
    for i in range(0, verticalpatches):
        for j in range(0, horizontalpatches):
            x0, x1 = i * dimpatch, (i + 1) * dimpatch
            y0, y1 = j * dimpatch, (j + 1) * dimpatch
            patches_list.append(extended[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    arrayofpatches = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(arrayofpatches, batch_size=4)
    print("PATHC PRDCT:",patches_predict.shape)
    prediction = np.zeros(shape=(addedh,addedw, numclass), dtype=np.float32)
    print("prediction shape:",prediction.shape)
    for k in range(patches_predict.shape[0]):
        i = k // horizontalpatches
        j = k % verticalpatches
        x0, x1 = i * dimpatch, (i + 1) * dimpatch
        y0, y1 = j * dimpatch, (j + 1) * dimpatch
        print("I:",i)
        print("J:",j)
        
        print("SZ:",dimpatch)
        print("X0,X1:",x0,x1)
        print("Y0,Y1:",y0,y1)
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:h, :w, :]

if __name__ == '__main__':
    model = get_model()
    model.load_weights(weights_path)
    img = normalize(tiff.imread('data/mband/test.tif'))
    print("IMG H",img.shape[0])
    print("IMG W",img.shape[1])
    print("IMG Channel",img.shape[2])
    #img = np.expand_dims(img,)
    xdimens = img.shape[0]
    ydimens = img.shape[1]
    s = (xdimens,ydimens)
    extrachannel1=np.zeros(s)
    extrachannel2=np.zeros(s)
    extrachannel3=np.zeros(s)   
    extrachannel4=np.zeros(s)
    extrachannel5=np.zeros(s)
    output = np.zeros((xdimens, ydimens, 8))
    output[:,:,0] = img[:,:,0]
    output[:,:,1] = img[:,:,1]
    output[:,:,2] = img[:,:,2]
    output[:,:,3] = extrachannel1
    output[:,:,4] = extrachannel2
    output[:,:,5] = extrachannel3
    output[:,:,6] = extrachannel4
    output[:,:,7] = extrachannel5
    img = output
            
    for i in range(7):
        if i == 0:  # reverse first dimension
            outmatrix = predictoutput(img[::-1,:,:], model, dimpatch=dimpatchsz, numclass=outputClasses).transpose([2,0,1])
        elif i == 1:    # reverse second dimension
            temp = predictoutput(img[:,::-1,:], model, dimpatch=dimpatchsz, numclass=outputClasses).transpose([2,0,1])
            outmatrix = np.mean( np.array([ temp[:,::-1,:], outmatrix ]), axis=0 )
        elif i == 2:    # transpose(interchange) first and second dimensions
            temp = predictoutput(img.transpose([1,0,2]), model, dimpatch=dimpatchsz, numclass=outputClasses).transpose([2,0,1])
            outmatrix = np.mean( np.array([ temp.transpose(0,2,1), outmatrix ]), axis=0 )
    
        elif i == 3:
            temp = predictoutput(np.rot90(img, 1), model, dimpatch=dimpatchsz, numclass=outputClasses)
            outmatrix = np.mean( np.array([ np.rot90(temp, -1).transpose([2,0,1]), outmatrix ]), axis=0 )
        elif i == 4:
            temp = predictoutput(np.rot90(img,2), model, dimpatch=dimpatchsz, numclass=outputClasses)
            outmatrix = np.mean( np.array([ np.rot90(temp,-2).transpose([2,0,1]), outmatrix ]), axis=0 )
        elif i == 5:
            temp = predictoutput(np.rot90(img,3), model, dimpatch=dimpatchsz, numclass=outputClasses)
            outmatrix = np.mean( np.array([ np.rot90(temp, -3).transpose(2,0,1), outmatrix ]), axis=0 )

        else:
            temp = predictoutput(img, model, dimpatch=dimpatchsz, numclass=outputClasses).transpose([2,0,1])
            outmatrix = np.mean( np.array([ temp, outmatrix ]), axis=0 )
    
    tiff.imsave('predictedoutput.tif', (255*outmatrix).astype('uint8'))
   