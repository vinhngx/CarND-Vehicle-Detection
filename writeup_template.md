# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier 
* Optionally, apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[cars]: ./examples/cars_examples.png
[noncars]: ./examples/noncars_examples.png


[image2]: ./examples/HOG_example.jpg

[sliding_window_1]: ./examples/sliding_window_1.png
[sliding_window_2]: ./examples/sliding_window_2.png

[heatmap]: ./examples/heatmap.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the IPython notebook  [extract-combined-features-and-train-classifiers.ipynb](./extract-combined-features-and-train-classifiers.ipynb).

*Exploring data sets*: I first read and explore the training data sets. Combining all the car and non-car examples from the suggested training data sets [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip), we've got 8792 car and 8968 non-car examples. A random selection of samples is presented below:


![cars][cars]
![noncars][noncars]

#### 2. Explain how you settled on your final choice of HOG parameters.

I extracted 3 types of features:

- HOG with the following parameters:

```python
colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
```

Example visualization of of HOG features are presented in the IPython notebook  [extract-combined-features-and-train-classifiers.ipynb](./extract-combined-features-and-train-classifiers.ipynb).

- Spatial color features with bin size = (32, 32)

- Color histogram features  with nbins=32, bins_range=(0, 256)

These are rather standard parameter settings suggested in the lectures. Since the classifiers trained on top of these features exhibit good accuracy (>99%), I have not explored other parameter settings (such as other bin sizes and color spaces). 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained several classifiers on top of the extracted features, as detailed in the "Training classifiers" section of the IPython notebook  [extract-combined-features-and-train-classifiers.ipynb](./extract-combined-features-and-train-classifiers.ipynb).

- The data (extracted features) is first splitted into train-test sets at 80%-20% ratio.

- A simple linear SVM with default parameters reached a test accuracy of 96.26%, which is promising. Training is also rather speedy, requiring only ~37 secs. 

- I then attemped training a non-linear SVM with default parameters, however, it required too much time for training a single non-linear SVM (few hundreds of seconds on a 8-core machine). Since we would surely have to carry out a grid-search to find the optimal paramters setting, this would become too expensvie and also resource-inefficient, considering sklearn implementation of the SVM is single-threaded.

- I then attemped an ensemble decision tree method, named Xgboost. Xgboost generally perform very well with minimal parameter tweaking. 

I first tried training a small ensemble with the following paramters:

```python
model = xgb.XGBClassifier(
 learning_rate =0.1,
 n_estimators=150,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.75,
 colsample_bytree=0.75,
 objective= 'binary:logistic',
 nthread=16,
 scale_pos_weight=1)
 ```
 
which reached a test accuracy of 98.99%, which is very promising. The training and validation log-loss and classification errors are plotted to inspect potential overfitting issues.
 
I then proceed to training a larger ensemble, by lowering the learning and submsapling rates, in order to reduce overfitting.
 
 ```python
 model2 = xgb.XGBClassifier(
 learning_rate = 0.05,
 n_estimators=1000,
 max_depth=4,
 min_child_weight=1,
 gamma=0,
 subsample=0.5,
 colsample_bytree=0.5,
 objective= 'binary:logistic',
 nthread=16,
 scale_pos_weight=1)
 ```
 
 This classifier reached a test accuracy of 99.21% and is selected for the next step.
 
 Xgboost is fully parallel, making very efficient use of the 8-core machine (16 threads).


### Sliding Window Search

This step is carried out in the IPython notebook [Sliding-Window-Search-all-features-withHistory](./Sliding-Window-Search-all-features-withHistory.ipynb). We first reimplement the same feature extraction pipeline as described in the above section.

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is detailed in the "sliding window search" section of the IPython notebook [Sliding-Window-Search-all-features-withHistory](./Sliding-Window-Search-all-features-withHistory.ipynb).

For each image at a specific scale, we first compute the HOG features for the whole image. A window of size 64 is then slided throughout the Region of Interest in the image ```img[ystart:ystop,:,:]```  at 75% percent overlap (```cells_per_step=2``` given each windows consists of 8 cells).

Then, the features for the window is extracted as follows:

- HOG features are extracted from the pre-computed whole-image HOG features.

- Spatial color features and color histogram features are extracted for the window.

We then employ a heat map to accumulate detection results at different scale. Finally, a threshold is applied on the heat map and connected component of this thresholded heat map is identified using `scipy.ndimage.measurements.label()`. Each separate connected component corresponds to a detected car. We empirically found 3 to be a good threshold for eliminating false positives.


Example detection images are provided in the "sliding window search" section of the IPython notebook [Sliding-Window-Search-all-features-withHistory](./Sliding-Window-Search-all-features-withHistory.ipynb).

![sliding_window_1][sliding_window_1]
![sliding_window_2][sliding_window_2]
![heatmap][heatmap]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

A full car detection pipeline is implemented in the "Full car-detection pipeline" section in the "sliding window search" section of the IPython notebook [Sliding-Window-Search-all-features-withHistory](./Sliding-Window-Search-all-features-withHistory.ipynb).


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_processed.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To smooth out detection results on video, we concatenate the detection results, i.e. detected boxes, of 6 consecutive frames, and apply a higher threshold to the heat map. Empirically, a threshold of 3x6, i.e. threshold for individual frame x #frames,  was found to work reasonably well to suppress false positives that were not consistently detected accross a few frames. 

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
The video shows both the original frame as well as the heatmap. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Several difficulties were observed during this project:
- If the classifier accuracy is not high enough, e.g. <99%, then the number of false positive will be quite high, given that there are many sliding windows. Here I have not taken measures to reduce the number of sliding windows needed to be processed.

- The Xgboost classifier is quite speedy, but it becomes expensive when processing long videos. It is possible to use a smaller ensemble to speed up detection.

- The feature detection step was done quite efficiently for HOG feature extraction, but spatial features and color histogram features were extracted frame-by-frame, potentially resulting in redudant computation. 

- Using neural network architectures such as YOLO and SSD could improve this work by doing both detection and classification in a single feed-forward pass.




