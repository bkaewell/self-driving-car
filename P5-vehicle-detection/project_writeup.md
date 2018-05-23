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
[image04]: ./output_images/search_window_scale2.5.jpg "search window example"
[image05]: ./output_images/car_notcar_hog_feature.jpg "HOG example"
[image06]: ./output_images/car_notcar_hog_feature.jpg "HOG example"
[image07]: ./output_images/car_notcar_hog_feature.jpg "HOG example"

---

### Histogram of Oriented Gradients (HOG)

I started by reading in all the [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) images from a labeled training dataset stored on Amazon Web Services.  There are 8,792 vehicle images and 8,968 non-vehicle images.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html) and the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/).  Here is an example of one of each of the vehicle and non-vehicle classes:


![alt text][image01]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  The source code for this step is contained in the fourth and sixth code cells of the IPython Jupyter notebook called `pipeline.ipynb`, which can be found [here](https://github.com/bkaewell/self-driving-car/blob/master/P5-vehicle-detection/pipeline.ipynb).

Here is an example using the `YCrCb` color space for channel 0 and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image02]


I experimented with various combinations of HOG parameters and obtained the final parameter values by examining the accuracy of my classifier.  This resulted in a HOG feature vector length of 5,292.

I trained a linear Support Vector Machine (SVM) classifier using a combination of color and gradient features extracted from the training dataset.  These features were concatenated and flattened into a 1-dimensional array then preprocessed by normalization with mean-center.  In addition, a labels vector was defined to support the binary classification model with “1” representing the vehicle class and “0” representing all non-vehicles.  In order to gauge how well the classifier was working, I shuffled and split my vehicle and non-vehicle data into a training and testing set with the `test_size` parameter set to 0.25.  After experimenting with color spaces and channels, I found that all channels of the YCrCb color space produced an excellent test accuracy score of 98.6%.

---

### Sliding Window Search

1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The source code for this step is contained in the tenth code cell of the notebook located in "pipeline.ipynb" in the function called `find_cars()`.

(...get help from lectures...)

To bound the search window region, I decided to divide the 1200 x 760 image in half along the horizontal plane separating the sky and ground.  Since self driving cars are not cruising in the skies now, I only processed the lower half of the image, starting at the 400 pixel mark.  I decided to search window positions at different scales ranging from 1 to 2.5 in 0.5 steps and came up with this:


![alt text][image03]


![alt text][image04]


2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

---

### Video Implementation

1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Here are six frames and their corresponding heatmaps:


![alt text][image5]


Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:


![alt text][image6]


Here the resulting bounding boxes are drawn onto the last frame in the series:


![alt text][image7]


---

### Discussion

1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.



