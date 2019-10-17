import random
import numpy as np

def get_rand_patch(inputImage, imgMask, sz=160):
    assert len(inputImage.shape) == 3 and inputImage.shape[0] > sz and inputImage.shape[1] > sz and inputImage.shape[0:2] == imgMask.shape[0:2]
    x_side = random.randint(0, inputImage.shape[0] - sz)
    y_side = random.randint(0, inputImage.shape[1] - sz)
    patch_inputImage = inputImage[x_side:(x_side + sz), y_side:(y_side + sz)]
    patch_imgMask = imgMask[x_side:(x_side + sz), y_side:(y_side + sz)]

    randTransform = np.random.randint(1,8)
    if randTransform == 1:  
        patch_inputImage = patch_inputImage[::-1,:,:]
        patch_imgMask = patch_imgMask[::-1,:,:]
    elif randTransform == 2:    
        patch_inputImage = patch_inputImage[:,::-1,:]
        patch_imgMask = patch_imgMask[:,::-1,:]
    elif randTransform == 3:    
        patch_inputImage = patch_inputImage.transpose([1,0,2])
        patch_imgMask = patch_imgMask.transpose([1,0,2])
    elif randTransform == 4:
        patch_inputImage = np.rot90(patch_inputImage, 1)
        patch_imgMask = np.rot90(patch_imgMask, 1)
    elif randTransform == 5:
        patch_inputImage = np.rot90(patch_inputImage, 2)
        patch_imgMask = np.rot90(patch_imgMask, 2)
    elif randTransform == 6:
        patch_inputImage = np.rot90(patch_inputImage, 3)
        patch_imgMask = np.rot90(patch_imgMask, 3)
    else:
        pass
    return patch_inputImage, patch_imgMask


def get_patches(x_dict, y_dict, n_patches, sz=160):
    numpatches = n_patches
    xlist = list()
    ylist = list()
    total = 0
    while total < numpatches:
        inputImage_id = random.sample(x_dict.keys(), 1)[0]
        inputImage = x_dict[inputImage_id]
        imgMask = y_dict[inputImage_id]
        inputImage_patch, imgMask_patch = get_rand_patch(inputImage, imgMask, sz)
        xlist.append(inputImage_patch)
        ylist.append(imgMask_patch)
        total += 1
    print('Generated {} patches'.format(total))
    return np.array(xlist), np.array(ylist)


