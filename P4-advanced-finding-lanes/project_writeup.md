## Advanced Lane Finding

### Using Computer Vision to Find Lanes on Roadways

---

Goals:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


[image01]: ./output_images/undistorted_cal.jpg "Undistorted Cal"
[image02]: ./output_images/undistorted.jpg "Undistorted"
[image03]: ./output_images/thresholded.jpg "Thresholded"
[image04]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image05]: ./output_images/color_lines_fit.jpg "Lines Fit Visual"
[image06]: ./output_images/color_lanes_fit.jpg "Lanes & Lines Fit Visual"
[image07]: ./output_images/color_lane_area.jpg "Output"
[video01]: ./output_video.mp4 "Video"


### Camera Calibration

The code for this step is contained in the third code cell of the IPython notebook located in "pipeline.ipynb" in the function called “camera_calibration”. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same
for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be
appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints`
will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful
chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and
distortion coefficients using the OpenCV function `cv2.calibrateCamera()`. I applied this distortion correction to the
test image using the OpenCV function `cv2.undistort()` and obtained this result:

![alt text][image01]


### Pipeline (single images)

The first step in analyzing images is to undo the distortion from the camera lens. This is necessary to extract
correct and useful information from images of the road such as lane curvature, traffic signs and other objects.
After calibrating and correcting for distortion on the raw camera images, I can accurately determine where my
self driving car is in the world. Here is an example of undistorting a camera image:

![alt text][image02]

There is a subtle difference between the two images.  The original image shows the license plate of the white car, but it is not visible in the undistorted image.  This effect is due to the curvature of the camera lens and it is very important for autonomous vehicles to accurately identify objects and their true positions to ensure safe driving.


I used a combination of color ("R" channel from RGB and "S" channel from HLS) and gradient thresholds to generate a thresholded binary image.  During one of the lab experiments, I discovered that the R channel works best for white lines and the S channel works best for yellow lines.  Since white and yellow lines are standard line markings for roadways, I wanted to build a robust pipeline to perform the thresholding under all conditions so I included both R and S channel thresholding techniques.  One improvement is to divide the image in half and use R and S channels for the left side since the left lines can be either white or yellow.  And the right side is primarily just white lines. The code for this step is contained in the fourth code cell of the IPython notebook located in "pipeline.ipynb" in the function called “thresholding”.   Here's an example of my output for this step:

![alt text][image03]


The code for my perspective transform includes a function called `warp()`, which appears in the fifth code cell of the IPython notebook located in "pipeline.ipynb".  The `warp()` function takes as inputs a thresholded binary image (`image`), as well as source (`src`) and destination (`dst`) points.  I chose to hardcode the source and destination points in the following manner:

```python

    #Source points - defined area of lane line edges
    src = np.float32([[690, 450], 
                      [1110, image.shape[0]], 
                      [175, image.shape[0]], 
                      [595, 450]])
    
    #Destination points to transform from source points
    offset = 300 # offset for dst points
    dst = np.float32([[image.shape[1]-offset, 0], 
                      [image.shape[1]-offset, image.shape[0]],
                      [offset, image.shape[0]], 
                      [offset, 0]]) 
```

This resulted in the following source and destination points:

| Source     | Destination  | 
|:----------:|:------------:| 
| 690, 450   | 980, 0       | 
| 1110, 720  | 980, 720     |
| 175, 720   | 300, 720     |
| 595, 450   | 300, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image that had straight lanes and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image04]








#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]






---

### Pipeline (video)

Here's a [link to my video result](https://github.com/bkaewell/self-driving-car/blob/master/P4-advanced-finding-lanes/output_video.mp4)

The lane finding performs very well even though there are some wobbly lines when the car bounces on a random bump, but there are no catastrophic failures that would cause the car to drive off the road!

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
