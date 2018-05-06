## Advanced Lane Finding

### Using Computer Vision to Find Lanes on Roadways

---

Goals:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
* Apply a distortion correction to raw images
* Use color transforms, gradients, etc., to create a thresholded binary image
* Apply a perspective transform to rectify thresholded binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary
* Determine the curvature of the lane and vehicle position with respect to center
* Warp the detected lane boundaries back onto the original image
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position

[//]: # (Image References)
[image01]: ./output_images/undistorted_cal.jpg "Undistorted Cal"
[image02]: ./output_images/undistorted.jpg "Undistorted"
[image03]: ./output_images/thresholded.jpg "Thresholded"
[image04]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image05]: ./output_images/color_lines_fit.jpg "Lines Fit Visual"
[image06]: ./output_images/color_lanes_fit.jpg "Lanes & Lines Fit Visual"
[image07]: ./output_images/color_lane_area.jpg "Output"
[video01]: ./output_video.mp4 "Video"

---

### Camera Calibration

The code for this step is contained in the third code cell of the IPython notebook located in "pipeline.ipynb" in the function called `camera_calibration()`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the OpenCV function `cv2.calibrateCamera()`. I applied this distortion correction to the test image using the OpenCV function `cv2.undistort()` and obtained this result:

![alt text][image01]

---

### Pipeline (single images)

The following pipeline steps are contained in a processing chain called `process_image()`.

The first step in analyzing images is to undo the distortion from the camera lens. This is necessary to extract
correct and useful information from images of the road such as lane curvature, traffic signs and other objects.
After calibrating and correcting for distortion on the raw camera images, I can accurately determine where my
self driving car is in the world. Here is an example of undistorting a camera test image:

![alt text][image02]

There is a subtle difference between the two images.  The original image shows the license plate of the white car, but it is not visible in the undistorted image.  This effect is due to the curvature of the camera lens and it is very important for autonomous vehicles to accurately identify objects and their true shapes, sizes, and positions to ensure safe driving.

I used a combination of color ("R" channel from RGB and "S" channel from HLS) and gradient (Sobel X) thresholds to generate a thresholded binary image.  During one of the lab experiments, I discovered that the R channel works best for white lines and the S channel works best for yellow lines.  Since white and yellow lines are standard line markings for roadways, I wanted to build a robust pipeline to perform the thresholding under all conditions so I included both R and S channel thresholding techniques.  One improvement is to divide the image in half and use R and S channels for the left side since the left lines can be either white or yellow.  The right side is primarily white lines. The code for this step is contained in the fourth code cell of the notebook in the function called `thresholding()`.   Here's an example of my output for this step:

![alt text][image03]


The code for my perspective transform includes a function called `warp()`, which appears in the fifth code cell of the  notebook.  The `warp()` function takes as inputs a thresholded binary image, as well as source `src` and destination `dst` points.  I chose to hardcode the source and destination points in the following manner:

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

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image that has straight lanes and its warped counterpart to verify that the lines appear parallel in the warped image.  See the figure below:

![alt text][image04]


Then I took a histogram along all the columns of the lower half of the image to identify the strongest two peaks, presumably the left and right lane lines.  These peaks are good indicators of the x-position of the base of the lane lines. I used that as a starting point for where to search for the line pixels. From that point, I used a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.  Once I have all of the "hot pixels" and their x-y positions, then I fit a 2nd order polynomial to them. The code for this step is contained in the sixth code cell of the notebook in the function called `detect_lane_pixels()`.

I skip the previous sliding window step after I establish the line locations since the base of the lane lines are consistent from frame to frame (see functions `update_line_class()` and `find_lane_boundary()`). I fit my new lane lines (smoothed and averaged over the previous 5 polynomial fits) with a 2nd order polynomial kinda like this: 

![alt text][image05]

![alt text][image06]


I calculated the radius of curvature of the lane and the position of the vehicle with respect to center in the function `calc_lane_curvature()`.      


Here is a visualization of my final result of the pipeline, which illustrates the superimposed lane guides (in green) on a test image for the autonomous vehicle:

![alt text][image07]


---

### Pipeline (video)

Here's a [link to my video result](https://github.com/bkaewell/self-driving-car/blob/master/P4-advanced-finding-lanes/output_video.mp4)

The lane finding performs very well even though there are some wobbly lines when the car bounces on a random bump, but there are no catastrophic failures that would cause the car to drive off the road!

---

### Discussion

My approach was to start with the most difficult test image, which is the image I used for this writeup above.  The combination of shadows, cars, curves, and bridge texture was adequate for determing final threshold parameters. I used trial and error to tune the threshold parameters.  One potential improvement would be to incorporate an adaptive threshold.  However, as an alternative to the adaptative threshold, I performed an OpenCV median blur on all images prior to thresholding.  This helped reduce the wobbly lines on the left lane line when the car approached the bridge and made a sharp right turn. I also experimented with moving averages and different history buffer depths, but just a regular mean-average did pretty well.  The pipeline could be improved with more error handling when there are no "hot pixels" and the line fitting class variables are empty for more challenging environments.
