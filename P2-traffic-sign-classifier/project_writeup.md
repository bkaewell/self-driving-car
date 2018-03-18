# **Traffic Sign Recognition with Deep Learning** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Classify traffic signs using a simple convolutional neural network
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/traffic_sign_histogram.jpg "Visualization"
[image2]: ./output/traffic_sign_examples.jpg "Traffic Signs"
[image3]: ./output/traffic_sign_normalized.jpg "Traffic Sign Normalized"

---
### Writeup / README

Here is a link to my [project code](https://github.com/bkaewell/self-driving-car/blob/master/P2-traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set.

I used the numpy, pandas, and python libraries to calculate summary statistics of the German traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Before I start building my deep learning network, here is an exploratory visualization of the data set that I used from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  The figure below is a histogram illustrating an uneven distribution in the data set.  There are many peaks and valleys that form majority and minority classes.  I believe a solid starting point to balance out the training data set would be to up-sample the minority classes by randomly duplicating observations.  This would reinforce the minority classes' signal while reducing potential bias in the model to the majority classes.  This could be an area of improvement for preprocessing the original data.

![alt text][image1]


The data set contains over 34,000 images divided into 43 different classes ranging from speed limit signs to wild animal crossings.  Here are a few examples selected at random:

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

After exploring a limited subset of the training data, I decided that the colors of the traffic signs would be a better learning experience for the neural network than converting it to grayscale.  As a first step, I decided to normalize the image data using the min-max scaling technique because I wanted to build a well conditioned model with zero mean and small variance to keep it very uncertain about things.  Furthermore, it makes it easier for the feed forward optimizer to proceed numerically (i.e. accelerates the convergence of the model to the solution of accurate classification).  

Here is an example of a traffic sign image before and after normalization:

![alt text][image3]

The difference between the original data set and the augmented data set is the dynamic range of the pixels due to normalization with min-max scaling to a pixel intensity range of (0.1, 0.9).  Notice the whites are not as white and the darks are not as dark producing a small variance pixel to pixel.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers resembling LeNet 5:

| Layer                 		| Dimensions      		|     Description	        						| 
|:-----------------------------:|:---------------------:|:-------------------------------------------------:| 
| Convolution Layer 1 5x5  		| Input: (32,32,3) 		| 32x32x3 RGB image input 							| 
| 								| Output: (14,14,6) 	| -2D Convolution Layer 							|
| 								|   					|     -1x1 stride, valid padding					|
| 								|   					| -RELU Activation									|
| 								|   					| -Dropout 0.75 (keep 75% of neurons) 				|
| 								|   					| -2D Max Pooling (down sample)						|
| 								|   					|     -2x2 stride, 2x2 kernel size, valid padding	|
| 								|   					| 													|
| Convolution Layer 2 5x5  		| Input: (14,14,6) 		| -2D Convolution Layer 							| 
| 								| Output: (5,5,16)  	|     -1x1 stride, valid padding					|
| 								|   					| -RELU Activation									|
| 								|   					| -Dropout 0.75 (keep 75% of neurons) 				|
| 								|   					| -2D Max Pooling (down sample)						|
| 								|   					|     -2x2 stride, 2x2 kernel size, valid padding	|
| 								|   					|													|
| Fully Connected Layer 3  		| Input: 400 			| -Linear (WX + b) 									|
| 								| Output: 120 			| -RELU Activation									| 
| 								|   					| -Dropout 0.75 (keep 75% of neurons) 				|
| 								|   					| 													|
| Fully Connected Layer 4  		| Input: 120 			| -Linear (WX + b) 									|
| 								| Output: 84 			| -RELU Activation									| 
| 								|   					| -Dropout 0.75 (keep 75% of neurons) 				|
| 								|   					| 													|
| Output Layer 5  				| Input: 84 			| -Linear (WX + b) 									|
| 								| Output: 43 			| 													|
| 								|   					| 													|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

