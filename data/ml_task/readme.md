
# README

## Content

- left/ contains RGB images.
- disp/ contains disparity images (see stereo assignment task). They are pretty noisy. The format is png 16 bit.
- label/ contains label images. The labels were manually drawn and are far from perfect.

The mapping is as follows:

{'building': 1,
 'dirt': 2,
 'foliage': 3,
 'grass': 4,
 'human': 5,
 'pole': 6,
 'rails': 7,
 'road': 8,
 'sign': 9,
 'sky': 10}
0 means unlabelled and can be ignored. 

## General task

Semantic segmentation: give a class label to each pixel.

## Variations on actual task

You are free to choose the exact task and how to solve it. 
In any case it is important to write up a good report explaining what you did
and why.

Desirable attributes of your solution(s) include:
- robustness
- originality
- efficiency
- accuracy
- thoroughness in evaluation

The balance between these attributes is up to you. Accuracy by itself is *not*
the goal - we would rather see a creative approach with a good report, rather
than a very accurate classifier that is exclusively using off-the-shelf code
with a bad report.

The suggested baseline is to extract image features (such as LBP, textons, HOG,
etc) for each pixel and use labels to train a simple classifier to label each
pixel in each frame.

Possible improvements beyond this:
- Use segmentation and/or superpixels before labelling.
- Model context with markov random field, conditional random field, hierarchical stacked labeling
- Use unlabelled data to improve results ("semisupervised" approach).
- Use temporal context between frames
- Use depth/disparity in an interesting way
- Use data from an external source, e.g. http://groups.csail.mit.edu/vision/SUN/ to improve results
- Anything else you can come up with.

## Output

- There should be quantitative metrics (Accuracy, Confusion matrices,
  Precision/Recall) *with crossvalidation*.
- There should be qualitiative results (examples) of success and of failure. A
  video is highly desirable.

## Notes

- This data is private, please do not share.
- We favor Python and C++ over Matlab as they integrates better with ROS and
  are open source.
- It's ok to use third party code for stuff like extracting superpixels or HOG
  features. On the other hand,
  if your solution is *just* third party code then that is not very impressive.

Feel free to ask questions to dimatura@cmu.edu
