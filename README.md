# Semantic_Perception
Semantic Perception internship assignment for CMU 2016 summer internships





The entire pipeline can be divided into four parts. Each part has a corresponding source file, and the roles of each are explained below:


## 1. features_slic (Segmentation, Feature Vector Extraction)
The first task is to segment all the images into superpixels and then extract feature vectors from each of those superpixels. This task is done recursively for all the files using the *scripts/FV_Extraction* script.:

* As an input, this executable requires the image number ('0000' or '0480', refering to the left images in the /data/ml_task/ folder).
* It then performs a SLIC segmentation of the image, which is a very powerful technique that only uses the color and position information of the pixels.
* Once we have the superpixels, the feature vectors for each superpixel are calculated and saved. These feature vectors are determined using properties like the average RBG values, Gaussian Gradients, Cornerness, etc. and are what get fed to the subsequent classifiers in the pipeline.


## 2. GT_SLICLabels_extraction (Extract Ground Truth Values and SLIc Labels)
This file extracts the ground truth labels (the desired outputs) for each feature vector using the labeled images. These are determined by taking the most frequent pixel-wise label occuring in each superpixel corresponding to each feature vector. The step is done for all of the training images using the *scripts/GT_SLICLabel_Extraction* script and the output of is stored in these folders:

1. *data/GT_vectors* - Each file in this folder lists the ground truth values for each superpixel in the form of a column vector - one vector per image, and one superpixel per entry.
2. *data/SLICLabel_vectors* - This contains the labels of the superpixels (which are numbers). The reason I had to output this explicitly is because of a small glitch in the VIGRA SLIC segmentation implementation.
3. *data/SLICSegmentation_arrays* - This contains arrays of the size of the images, wherein each pixel is labeled with the label of the superpixel it belongs to.


## 3. kNN (k - Nearest Neighbours to Assign Labels to Feature Vectors)
A particular training-testing division is chosen (37 images training, 13 images testing) and the feature vectors of the superpixels in the training images are used as reference points for the k-NN algorithm. The 5 nearest neighbors of the testing feature vectors are used to determine what their labels should be. One way to proceed is this:

* The label of the feature vector is predicted by assigning scores to all of the possible labels of the 5 nearest neighbours against the inverse distances from those neighbours - the label with the highest score is then chosen as the label of the feature vector.

But, the algorithm used in this pipeline further transforms this vector of scores (using extra information extracted from the training vectors) before predicting the label of the vector to be the one with the maximum score. The exact algorithm for this is explained below:

1. The extraction of extra information from the training vectors begins by finding the 4 nearest neighbours for each of the training vectors (the first nearest neighbour would be the training vector itself, which is part of the set of reference points for the algorithm). 
2. These 4 nearest neighbours are used to determine the expected labels for the training vectors (using the naive algorithm described above).

## 4. eval (Quantify Pipeline Performance)
This part of the pipeline takes the predicted images and compares them pixel-wise with the original labeled (ground truth) images to calculate the accuracy (confusion matrix, precision, recall and F1) of the pipeline.