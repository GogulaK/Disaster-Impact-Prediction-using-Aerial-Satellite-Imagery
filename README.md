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
* The following features are more relevant for the disaster impact prediction task:
○ Buildings
○ Roads
○ Trees
○ Water
○ Crops

## Requirements
Create a conda environment with the following dependancies:
* tiff file
* tensorflow
* opencv2
* Keras
* numpy
