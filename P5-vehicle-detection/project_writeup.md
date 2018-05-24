## Vehicle Detection & Tracking

### Using Computer Vision and Machine Learning to Detect and Track Vehicles on Roadways

---

Goals:

 * Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
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

To bound the search window region, I decided to divide the 1280 x 720 image in half along the horizontal plane separating the sky and ground.  Since there is no altitude dimension to self-driving cars today, I only processed the lower half of the image, starting at the 400 pixel mark.  A visualization of small overlapping windows near the horizon is shown below with a blue swath and a green swath of windows:


![alt text][image03]


As the scale factor increases, the search area of the windows increases, but the total number of windows decreases.  Here is a visualization of large overlapping windows:


![alt text][image04]


Ultimately I searched on four scales (1, 1.5x, 2x, and 3x) with two swaths per scale and overlapping windows 50% in X and 75% in Y directions.  I used YCrCb 3-channel HOG features plus histograms of color in the feature vector, which provided a solid result.  Here are some example images of the full sliding window processing:

![alt text][image05]


I optimized the performance of my classifier by adding a confidence score threshold to minimize the false positive detections from the classifier....

![alt text][image06]

---

### Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](https://github.com/bkaewell/self-driving-car/blob/master/P5-vehicle-detection/output_video.mp4)


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

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

-i noticed i had a consistent vehicle detection on the far right side of the frame for a 2.5x scale factor for the duration of the video.   

-i noticed that i wasn't picking up any of the smaller cars on the road with my current scale factors.  a potential area of improvement is to reduce the window size by setting the scale factor below 1 and experiment with that.

-area of improvement smoothing out the transition 

-trade offs between minimizing false positives, maybe did that too much.. what about oncoming traffic? or a parked car? 
-using radar as an additional sensor for detection