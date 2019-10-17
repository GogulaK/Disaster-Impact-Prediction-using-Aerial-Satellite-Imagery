from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping


def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

outputClasses = 5  
numbands = 8
outputClassWeight = [0.2, 0.3, 0.1, 0.1, 0.3]
dimpatchsz = 160   
BatchSize = 150
trainSize = 4000  
validationSize = 1000    
EPOCHS = 150


def get_model():
    return unet_model(outputClasses, dimpatchsz, n_channels=numbands, upconv=True, class_weights=outputClassWeight)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(2) for i in range(1, 25)]  

if __name__ == '__main__':
    x_train = dict()
    y_train = dict()
    x_train_validation = dict()
    y_train_validation = dict()

    print('Reading images')
    for imageId in trainIds:
        img_modified = normalize(tiff.imread('./data/mband/{}.tif'.format(imageId)).transpose([1, 2, 0]))
        mask = tiff.imread('./data/gt_mband/{}.tif'.format(imageId)).transpose([1, 2, 0]) / 255
        train_x_size = int(3/4 * img_modified.shape[0])  # use 75% of image as train and 25% for validation
        x_train[imageId] = img_modified[:train_x_size, :, :]
        y_train[imageId] = mask[:train_x_size, :, :]
        x_train_validation[imageId] = img_modified[train_x_size:, :, :]
        y_train_validation[imageId] = mask[train_x_size:, :, :]
    
    def train_net():
        print("start train net")
        x_train, y_train = get_patches(x_train, y_train, n_patches=trainSize, sz=dimpatchsz)
        x_val, y_val = get_patches(x_train_validation, y_train_validation, n_patches=validationSize, sz=dimpatchsz)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
        model.fit(x_train, y_train, batchSize=BatchSize, epochs=EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net()
