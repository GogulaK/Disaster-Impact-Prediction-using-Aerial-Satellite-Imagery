# Disaster-Impact-Prediction-using-Aerial-Satellite-Imagery

## About this project
* This is a Keras based implementation to measure disaster impact from aerial satellite imagery using Unet architecture.  
* We perform CNN-based semantic segmentation on satellite imagery obtained before and after disaster to identify areas of maximal damage.
* Our work incorporates high-level features including roads,buildings,trees,crops and water. This renders increased flexibility to proposed model over existing approaches.
* Our experiments show a high correlation between predicted impact areas and human –annotated ground truth labels.

## Dataset
* The dataset for this project is adapted from a subset of the Spacenet dataset.
* Low resolution M-band images are used.
* M band(400-1040 nm), 1.24m/pixel, color depth 11 bit.
* The following features are more relevant for the disaster impact prediction task: Buildings, Roads, Trees, Water, Crops.

## Requirements
Create a conda environment with the following dependancies:
* tiff file
* tensorflow
* opencv2
* Keras
* numpy

## Running the Code
Preparing data
* Add the test image to the data/mband/ folder
* Rename the file to be tested as test.tif in /data/mband folder
* Make sure the image is square in dimensions
* The result is stored as predicted output.tif
* There will be a series of 5 images in tif
* Each one represents a feature as per code (eg: 3 is buildings)

Run python predict.py

Perform the same for pre disaster and post disaster image separately.
* Store the predicted output of predisaster as result.tif
* Store the predicted output of post disaster as postresult.tif

Run Python postprocess.py
* This produces the aggregated presummed.tif and postsummed.tif, diffprepost.tif
* Copy the diffprepost.tif,summedpre.tif,summedpost.tif to a new folder (eval_folder)

Run python heatmap_to_tif.py 'Foldername' #Run it from the previous folder  
* The heatmap will be stored in the given Foldername.

Run python evaluation.py
* evaluation.py calculates F1_score and the steps are as follows,

Input to evaluation.py are: diffprepost.tif , corresponding ground truth label.csv (available in sample) 
        
Note: Metric used => Disaster Impact Index(DII) = |ηPredbefore=1&Predafter=0|grid /  Ngrid |ηPredbefore=1 for entire image|      

Based on above metric, Labels will be generated for predicted outputs and the same would be compared against ground truth label(uses the same metric)
     


