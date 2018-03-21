# **Traffic Sign Classification with Deep Learning**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Classify traffic signs using a simple Convolutional Neural Network (CNN)
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/traffic_sign_histogram.jpg "Visualization"
[image2]: ./output/traffic_sign_examples.jpg "Traffic Signs Examples"
[image3]: ./output/traffic_sign_normalized.jpg "Traffic Sign Normalized"

[image4]: ./new_test_images/test1.jpg "Traffic Sign Test 1"
[image5]: ./new_test_images/test2.jpg "Traffic Sign Test 2"
[image6]: ./new_test_images/test3.jpg "Traffic Sign Test 3"
[image7]: ./new_test_images/test4.jpg "Traffic Sign Test 4"
[image8]: ./new_test_images/test5.jpg "Traffic Sign Test 5"


---
### Writeup / README

Here is a link to my [project code](https://github.com/bkaewell/self-driving-car/blob/master/P2-traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary

I used the numpy, pandas, and python libraries to calculate summary statistics of the German traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory Visualization

Before I start building my deep learning network, here is an exploratory visualization of the data set that I used from the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  The figure below is a histogram illustrating an uneven distribution in the data set.  There are many peaks and valleys that form majority and minority classes.  I believe a solid starting point to balance out the training data set would be to up-sample the minority classes by randomly duplicating observations.  This would reinforce the minority classes' signal while reducing potential bias in the model to the majority classes.  This could be an area of improvement for preprocessing the original data.


![alt text][image1]


The data set contains over 50,000 images divided into 43 different classes ranging from speed limit signs to wild animal crossings.  Here are a few examples selected at random:


![alt text][image2]


### Preprocessing

After exploring a limited subset of the training data, I decided that the colors of the traffic signs would be a better learning experience for the neural network than converting it to grayscale.  As a first step, I decided to normalize the image data using the min-max scaling technique because I wanted to build a well conditioned model with zero mean and small variance to keep it very uncertain about things.  Furthermore, it makes it easier for the feed forward optimizer to proceed numerically (i.e. accelerates the convergence of the model to the solution of accurate classification).  

Here is an example of a traffic sign image before and after normalization:


![alt text][image3]


The difference between the original data set and the augmented data set is the dynamic range of the pixels due to normalization with min-max scaling to a pixel intensity range of (0.1, 0.9).  Notice the whites are not as white and the darks are not as dark producing a small variance pixel to pixel.


### Model Architecture

My final model architecture is a multi-layer CNN to classify the traffic signs from Germany using TensorFlow.  It consisted of the following layers resembling the "LeNet-5" function:

|Layer						|Dimension			|Description															| 
|:-------------------------:|:-----------------:|:---------------------------------------------------------------------:| 
|Convolution Layer 1 (5x5)  |Input: (32,32,3)   |32x32x3 RGB image input -> ...                                         | 
|							|Output: (14,14,6)  |2D Convolution Layer -> RELU Activation -> Dropout -> 2D Max Pooling   |
|							|					|																		|
|							|					|																		| 
|Convolution Layer 2 (5x5)  |Input: (14,14,6)   |2D Convolution Layer -> RELU Activation -> Dropout -> 2D Max Pooling   |
|							|Output: (5,5,16)   |																		|
|							|					|																		|
|							|					|																		|
|Fully Connected Layer 3    |Input: 400         |Flatten input -> ...                                                   |       
|							|Output: 120        |Linear (WX + b) -> RELU Activation -> Dropout                          | 
|							|					|																		|
|							|					|																		|
|Fully Connected Layer 4    |Input: 120         |Linear (WX + b) -> RELU Activation -> Dropout                          |
|							|Output: 84         |																		| 
|							|					|																		| 
|							|					|																		| 
|Output Layer 5             |Input: 84          |Linear (WX + b)														|
|							|Output: 43         |																		| 
|							|					|																		| 
|							|					|																		|

Weights for the above CNN are randomized from a normal distribution with zero mean and equal variance.  This prevents the model from getting stuck every time I train it.  Bias vector is set to zero.  These parameters are shared across all layers of the CNN.  

First layer is a CNN with a patch size of 5x5, a stride of 1x1, VALID padding and a depth of 6.  It uses a standard RELU activation function and dropout. I then applied a max pooling technique to down sample the output with a 2x2 stride, 2x2 patch/filter size and VALID padding.  The effect of down sampling is evident when comparing the input and output size in the dimension column in the table above.

Second layer is also a CNN with a patch size of 5x5, a stride of 1x1, VALID padding and a depth of 16.  It uses a standard RELU activation function and dropout. I then applied a max pooling technique to down sample the output with a 2x2 stride, 2x2 patch/filter size and VALID padding.

Third and fourth layers are fully connected layers with a width of 120 and 84, respectively.  Both use a standard RELU activation function and dropout.

The final layer, the output, is a fully connected layer with a width of 43 (total classes).


### Model Training

To train the model, I used an Adam Optimizer with the default parameter settings.  After numerous trials, I used a batch size of 100 and 100 for the number of training epochs.  I used a learning rate of 0.001.  The dropout rate for all layers (convolutional and fully connected) had a keep probability of 0.75 meaning 75% of the neurons were retained.


### Results

My final model results were:
* training set accuracy of ???
* validation set accuracy of ???
* test set accuracy of ???

To achieve my goal of 95% validation accuracy, I focused on parameter tuning and created a rough table of key parameters to test.  For example, Normalize vs Min-Max Scaling, no Dropout vs Dropout (with different keep probabilities), batch size, training epochs, and learning rate.  This plan adjusts and tests a little bit of everything including preprocessing, regularization, and model training to build a successful and accurate model.  


### Solution Approach

I used a CNN architecture called "LeNet-5" and implemented it in TensorFlow. I decided this was suitable for the current problem because the CNN learns to recognize basic lines and curves, then shapes and color blobs, and then increasingly complex objects within the image.  It is a very powerful method to learn and classify images similar to how humans perceive images.  In this traffic sign case, the levels of hierarchy are:

* Lines and curves
* Blobs of color, simple shapes, like circles and triangles
* Complex objects (combinations of simple shapes), like pedestrians, animals, and  cars
* The traffic sign as a whole (a combination of complex objects)


???
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
???

-Model evaluation:
-Evaluate how well the loss and accuracy of the model for a given data set
-loss/cost?


If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.


* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:



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


