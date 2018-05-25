## Vehicle Detection & Tracking

### Using Computer Vision and Machine Learning to Detect and Track Vehicles on Roadways

---

Goals:

 * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier using sci-kit learn: machine learning in Python
 * Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector
 * Normalize features and randomize a selection for training and testing
 * Implement a sliding-window technique and use your trained classifier to search for vehicles in images
 * Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles
 * Estimate a bounding box for vehicles detected

[//]: # (Image References)
[image01]: ./output_images/car_notcar_example.jpg "car/not car example"
[image02]: ./output_images/car_notcar_hog_feature.jpg "HOG example"
[image03]: ./output_images/search_window_scale1.0.jpg "search window example"
[image04]: ./output_images/search_window_scale3.0.jpg "search window example"
[image05]: ./output_images/sliding_window_examples.jpg "sliding window example"
[image06]: ./output_images/score_distribution_model_ref.png "Score Distribution Example"

[image07]: ./output_images/car_notcar_hog_feature.jpg "HOG example"
[image08]: ./output_images/car_notcar_hog_feature.jpg "HOG example"
[image09]: ./output_images/car_notcar_hog_feature.jpg "HOG example"

---

### Histogram of Oriented Gradients (HOG)

I started by reading in all the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images from a labeled training dataset stored on Amazon Web Services.  There are 8,792 vehicle images and 8,968 non-vehicle images.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).  Here is an example of one of each of the vehicle and non-vehicle classes:


![alt text][image01]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  The source code for this step is contained in the fourth and sixth code cells of the IPython Jupyter notebook called `pipeline.ipynb`, which can be found [here](https://github.com/bkaewell/self-driving-car/blob/master/P5-vehicle-detection/pipeline.ipynb).

Here is an example using the `YCrCb` color space for channel 0 and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image02]


I experimented with various combinations of HOG parameters and obtained the final parameter values by examining the accuracy of my classifier.  This resulted in a HOG feature vector length of 5,292.

I trained a linear Support Vector Machine (SVM) classifier using a combination of color and gradient features extracted from the training dataset.  These features were concatenated and flattened into a 1-dimensional array then preprocessed by normalization with mean-center.  In addition, a labels vector was defined to support the binary classification model with “1” representing the vehicle class and “0” representing all non-vehicles.  In order to gauge how well the classifier was working, I shuffled and split my vehicle and non-vehicle data into a training and testing set with `test_size` set to 0.25.  After experimenting with color spaces and channels, I found that all channels of the YCrCb color space produced an excellent test accuracy score of 98.6%.

---

### Sliding Window Search

The sliding window search is contained in the tenth code cell of my notebook located in "pipeline.ipynb" in the function called `find_cars()`.  This single function is the workhorse of my processing pipeline.  It is able to extract features using HOG sub-sampling and make predictions.  It only extracts hog features once on a sub-region of the image (defined by start and stop Y positions), for each of a small set of predetermined window sizes (defined by a scale argument), and then sub-sampled to obtain all of its overlaying windows.  Each window is defined by a scaling factor that impacts the window size.  The scale factor can be set on different regions of the image (e.g. smaller near the horizon, larger in the center).  

To bound the search window region, I decided to divide the 1280 x 720 image in half along the horizontal plane separating the sky and ground.  Since there is no altitude dimension to self-driving cars today, I only processed the lower half of the image, starting at the 400 pixel mark.  A visualization of small overlapping windows near the horizon is shown below with a blue and green swath of windows:


![alt text][image03]


As the scale factor increases, the search area of the windows increases, but the total number of windows decreases.  Here is a visualization of large overlapping windows:


![alt text][image04]


Ultimately I searched on four scales (1, 1.5x, 2x, and 3x) with two swaths per scale and overlapping windows 50% in X and 75% in Y directions for expanded coverage.  I used YCrCb 3-channel HOG features plus histograms of color in the feature vector, which provided a solid result.  Here are some example images of the full sliding window processing:

![alt text][image05]


I improved the reliability of my classifier by selecting a classification threshold using the `decision_function()` from scikit-learn SVM and comparing prediction scores against it.  Any observations with scores higher than the threshold are then predicted as the positive class (vehicles) and scores lower than the threshold are predicted as the negative class (non-vehicles).  After fine-tuning the threshold value (`classification_thresh=1`), the model predicted fewer false positives and obtained more reliable car detections (see bottom of `find_cars()` function).  The figure below from the [Amazon Machine Learning Developer’s Guide]( https://docs.aws.amazon.com/machine-learning/latest/dg/binary-classification.html) illustrates a threshold cut off line on a sample score distribution for a binary classification model.


![alt text][image06]


---

### Video Implementation

Here's a [link to my video output](https://github.com/bkaewell/self-driving-car/blob/master/P5-vehicle-detection/output_video.mp4)


2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  The source code is contained in the Nth code cell of my notebook located in "pipeline.ipynb" in the function called `process_image()`.





Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:


![alt text][image07]


Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:


![alt text][image08]


Here the resulting bounding boxes are drawn onto the last frame in the series:


![alt text][image09]


---

### Discussion

I decided to go with an application where I wanted my machine learning model to be extremely sure about the positive predictions actually being positive (high precision) and be able to afford to misclassify some positive examples as negative (moderate recall).  This approach worked out very well because the model produced no false positives and detected and tracked the true positives in a steady state throughout the entire video.  I encountered several problems but one noteworthy problem was I had a steady false positive at a fixed position of each frame on the far right side for a 2.5x scale factor.  This was a concern because the frame contained only the road, but there may have been a glare from the windshield causing the reoccurring false positive.  As a work-around, I just removed all 2.5x scale windows from my pipeline.  To further improve my pipeline, I would derive an optimal grid pattern for the sliding window search to detect cars on the horizon.  I would use a scale factor less than 1.  And if adding hardware is an option, I would use data fusion with a radar sensor to significantly improve performance in low light or poor visibility weather conditions.  I look forward to running my processing pipeline on more challenging videos to see how well it performs in different environments!

---