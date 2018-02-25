# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./test_images_output/hough_lines_solidYellowCurve.jpg "Hough Lines"
[image2]: ./test_images_output/ext_hough_lines_solidYellowCurve.jpg "Extrapolated Hough Lines"


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw lines() function.

In the Finding Lanes project, I implemented a pipeline in Python to detect the lane markings on a highway from a video stream.  I used computer vision techniques and tools including matplotlib, numpy, and cv2 from OpenCV to process the images.  

My pipeline consisted of 5 steps.  

First, I converted the input image to 8-bit grayscale, then applied Gaussian smoothing.  I found that setting the parameter of the kernel size to 5 was sufficient to filter out any possible noise in the image.  

After smoothing, I applied the canny transform and tuned the lower and upper pixel thresholds to detect the strongest edges (or strongest gradients) in the image.  Since the image was 8-bits, each pixel can take 2^8 (256) possible values.  I programmed the lower and upper threshold pixel values to be 60 and 180, respectively, honoring the recommended ratio of 1:3.  

Next I constructed a quadrilateral image mask to filter out the detected strong edges in other areas of the image.  The dimensions of the mask were based on perspective geometry where the parallel lines in object space converge to a vanishing point in the image.  In other words, the lane marking separation is wider closest to the camera and becomes gradually narrower furthest from the camera.  

After masking the image, I defined the Hough transform parameters and applied them on the masked edge detected image using the houghLinesP() API call.  An example of combining the Hough lines with the original image can be seen below:

![alt text][image1]

In the last step of my pipeline, I iterated over the Hough line segments for additional processing.  In order to draw a single line on the left and right lanes, I modified the draw lines() function by separating the line segments by their slope polarity to determine which segments are part of the left lane vs right lane. Then I took the average slope for both left and right lanes to draw a line from the top of the mask through the bottom of the image using the line equation formula.  I built in some error handling cases to prevent divide by zeros for cases when there was no slope.  An example of combining the extrapolated Hough lines with the original image can be seen below: 

![alt text][image2]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when there is a bend in the road and the vanishing point of the mask would need to be dynamically modified.   

Another shortcoming could be heavy traffic or a big truck covering the dotted lane line markings, which would temporarily confuse the pipeline.  


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to dynamically size the image mask.

Another potential improvement could be to use a Kalman filter to improve the accuracy of the predicted lane line markings.
