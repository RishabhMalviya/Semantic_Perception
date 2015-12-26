The entire pipeline can be divided into four parts. Each part has a corresponding source file, and the roles of each are explained below. Please note that the scripts must be run after cd-ing into the */scripts/* folder, and the individual executables must be run after cd-ing into the */build/* folder (after running CMake and Make):


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

But, the algorithm used in this pipeline further transforms this vector of scores (called a *confidence vector* using extra information extracted from the training vectors. It then predicts the label of the vector to be the one with the maximum score. The exact algorithm for this is explained below:

1. The extraction of extra information from the training vectors begins by finding the 4 nearest neighbours for each of the training vectors (the first nearest neighbour would be the training vector itself, which is part of the set of reference points for the algorithm). 
2. These 4 nearest neighbours are used to determine the expected labels for the training vectors (using the naive algorithm described above). These expected labels are used with the real labels to create a confusion matrix for the training vectors. These are what the */data/Predictions_n/Performance_Evaluation/intermediateConfusionMatrix.csv* files hold.
3. This confusion matrix is then transformed into a conditional probability matrix where the (i,j) entry of the matrix denotes the probability that the actual label (GroundTruth value) is 'i', given that the kNN search predicted the label 'j'. These are what the */data/Predictions_n/Performance_Evaluation/condProb.csv* files hold.
4. The kNN search is then run for the testing vectors and confidence vectors are obtained.
5. These confidence vectors are then transformed using the conditional probability matrix. That is, the ith entry of the confidence vector now becomes the dot product of the ith row of the conditional probability matrix with the original confidence vector.
6. Now, the index with the maximum value in the transformed confidence vector is assigned as its label.

NOTE: The number of nearest neighbours to consider in the various parts of the algorithm can be varied, though in the preceding discussion we assumed fixed numerical values.
NOTE: To run the algorithm without the added modifications, comment the for loop starting at line 449 of the *kNN.cpp* file and uncomment the line just after it - line number 452.

## 4. eval (Quantify Pipeline Performance)
This script *evaluations* takes the predicted images and compares them pixel-wise with the original labeled (ground truth) images to calculate the accuracy (confusion matrix, precision, recall and F1) of the pipeline. 

Another approach is to make the comparison superpixel-wise, and there is a separate script called *evaluations_superpixel* for doing this. As would be expected, the reported accuracy is much higher for this superpixel-wise comparison; but it is not really a correct representation of the accuracy of the pipeline, as the final task is to label the pixels themselves.

There was a minute increase in 'pixel-wise' performance thanks to the introduction of the explained modification in the kNN search algorithm. 
