# Semantic_Perception
Semantic Perception internship assignment for CMU 2016 summer internships





The entire pipeline can be divided into four parts. Each part has a corresponding source file, and the roles of each are explained below:


## 1. features_slic (Segmentation, Feature Vector Extraction)
The first task is to segment all the images into superpixels and then extract feature vectors from each of those superpixels:

* As an input, this executable requires the image number ('0000' or '0480', refering to the left images in the /data/ml_task/ folder).
* It then performs a SLIC segmentation of the image, which is a very powerful technique that only uses the color and position information of the pixels.
* Once we have the superpixels, the feature vectors for each superpixel are calculated and saved. These feature vectors are determined using properties like the average RBG values, Gaussian Gradients, Cornerness, etc. and are what get fed to the subsequent classifiers in the pipeline.


## 2. GT_SLICLabels_extraction (Extract Ground Truth Values and SLIc Labels)
This file extracts the ground truth labels (the desired outputs) for each feature vector using the labeled images. These are determined by taking the most frequent pixel-wise label occuring in the superpixel corresponding to the feature vector. The output of this step is stored in these folders:

  1. *data/GT_vectors* - Each file in this folder lists the ground truth values for each superpixel in the form of a column vector - one vector per image, and one superpixel per entry.
  2. *data/SLICLabel_vectors* - This contains the labels of the superpixels (which are numbers). The reason I had to output this explicitly is because of a small glitch in the VIGRA SLIC segmentation implementation.
  3. *data/SLICSegmentation_arrays* - This contains arrays of the size of the images, wherein each pixel is labeled with the label of the superpixel it belongs to.


## 3. kNN (Use k Nearest Neighbours to Assign Labels to Feature Vectors)


## 4. eval (Quantify Pipeline Performance)
This part of the pipeline takes the predicted images and compares them pixel-wise with the original labeled (ground truth) images to calculate the accuracy (confusion matrix, precision, recall and F1) of the pipeline.
