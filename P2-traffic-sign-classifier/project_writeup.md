# **Traffic Sign Classification with Deep Learning**

---

You can find my project code [here](https://github.com/bkaewell/self-driving-car/blob/master/P2-traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)


### Goals
* Classify traffic signs using a simple Convolutional Neural Network (CNN)
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./output/traffic_sign_histogram.jpg "Visualization"
[image2]: ./output/traffic_sign_examples.jpg "Traffic Signs Examples"
[image3]: ./output/traffic_sign_normalized.jpg "Traffic Sign Normalized"
[image4]: ./output/traffic_sign_model_accuracy.jpg "Model Accuracy"
[image5]: ./output/traffic_sign_loss_accuracy.jpg "Cross Entropy"
[image6]: ./traffic_sign_web_data/glatteis_gefahr_cropped_resized.jpg "Traffic Sign Test 1"
[image7]: ./traffic_sign_web_data/speed_limit_80_cropped_resized.jpg "Traffic Sign Test 2"
[image8]: ./traffic_sign_web_data/stoppschild_cropped_resized.jpg "Traffic Sign Test 3"
[image9]: ./traffic_sign_web_data/strassenbauarbeiten_cropped_resized.jpg "Traffic Sign Test 4"
[image10]: ./traffic_sign_web_data/uberholverbot_cropped_resized.jpg "Traffic Sign Test 5"
[image11]:  ./traffic_sign_web_data/wild_wechsel_cropped_resized.jpg "Traffic Sign Test 6"


---
### Writeup / README



## Data Set Exploration

### Data Set Summary

I used the numpy, pandas, and python libraries to calculate summary statistics of the German traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

### Exploratory Visualization

Before I start building my deep learning neural network, here is an exploratory visualization of the data set that I used from the [German Traffic Sign Recognition Benchmark website](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).  The figure below is a histogram illustrating an uneven distribution in the data set.  There are many peaks and valleys that form majority and minority classes.  I believe an area of improvement would be to balance out the training data set by up-sampling the minority classes.  One approach is randomly duplicating observations.  This would reinforce the minority classes' signal while reducing potential bias in the model towards the majority classes.


![alt text][image1]


The data set contains over 50,000 images divided into 43 different classes ranging from speed limit signs to wild animal crossings.  Here are a few examples selected at random:


![alt text][image2]


## Design and Test a Model Architecture

### Preprocessing

After exploring a limited subset of the training data, I decided that the traffic sign colors would be relevant so I did not convert to grayscale.  It just seems like blobs of color especially for a red stop sign could probably aid the learning experience of a neural network.  As a first step, I decided to normalize the image data using the min-max scaling technique because I wanted to build a well-conditioned model with zero mean and small variance for the RGB pixels to keep the model very uncertain about things.  Furthermore, it makes it easier for the optimizer to proceed numerically (faster searches to reach a solution).  

Here is an example of a traffic sign image before and after min-max scaling normalization:


![alt text][image3]


The difference between the original data set and the preprocessed data set is the dynamic range of the pixels due to normalization with min-max scaling to a pixel intensity range of (0.1, 0.9).  Notice the whites are not as white and the darks are not as dark producing a small variance pixel to pixel.


### Model Architecture

My final model architecture is a multi-layer CNN to classify the traffic signs from Germany using TensorFlow.  It consisted of the following layers resembling the "LeNet-5" function:

|Layer						|Dimension			|Description												| 
|:-------------------------:|:-----------------:|:---------------------------------------------------------:| 
|Convolution Layer 1 (5x5)  |Input: (32,32,3)   |32x32x3 RGB image input -> ...                             | 
|							|Output: (14,14,6)  |2D Convolution Layer -> RELU Activation -> 2D Max Pooling  |
|							|					|															| 
|Convolution Layer 2 (5x5)  |Input: (14,14,6)   |2D Convolution Layer -> RELU Activation -> 2D Max Pooling  |
|							|Output: (5,5,16)   |															|
|							|					|															|
|Fully Connected Layer 3    |Input: 400         |Flatten input -> ...                                       |       
|							|Output: 120        |Linear (WX + b) -> RELU Activation -> Dropout              | 
|							|					|															|
|Fully Connected Layer 4    |Input: 120         |Linear (WX + b) -> RELU Activation -> Dropout              |
|							|Output: 84         |															| 
|							|					|															| 
|Output Layer 5             |Input: 84          |Linear (WX + b)											|
|							|Output: 43         |															| 
|							|					|															| 

Weights for the above CNN are randomized from a normal distribution with zero mean and equal variance.  This prevents the model from getting stuck every time I train it.  Bias vector is set to zero.  These parameters are shared across all layers of the CNN.  

First layer is a CNN with a patch size of 5x5, a stride of 1x1, VALID padding and a depth of 6.  It uses a standard RELU activation function. I then applied a max pooling technique to down sample the output with a 2x2 stride, 2x2 patch size and VALID padding.  The effect of down sampling is evident when comparing the input and output size in the dimension column in the table above.

Second layer is also a CNN with a patch size of 5x5, a stride of 1x1, VALID padding and a depth of 16.  It uses a standard RELU activation function. I then applied a max pooling technique to down sample the output with a 2x2 stride, 2x2 patch size and VALID padding.

Third and fourth layers are fully connected layers with a width of 120 and 84, respectively.  Both layers use a standard RELU activation function and dropout.

The final layer, the output, is a fully connected layer with a width of 43 (total classes).


### Model Training

To train the model, I used an Adam Optimizer with the default parameter settings.  After numerous trials, I used a batch size of 100 and 100 for the number of training epochs.  I used a learning rate of 0.001.  The dropout rate for the fully connected had a keep probability of 0.75 meaning 75% of the neurons were retained.


### Results

My final model results were:
* training set accuracy of ???
* validation set accuracy of ???
* test set accuracy of ???


### Solution Approach

I used a CNN architecture called "LeNet-5" and implemented it in TensorFlow. I decided this was suitable for the current problem because the CNN learns to recognize basic lines and curves, then shapes and color blobs, and then increasingly complex objects within the image.  It is a very powerful method to learn and classify images similar to how humans perceive images.  In this traffic sign case, the levels of hierarchy are:

* Lines and curves
* Blobs of color, simple shapes, like circles and triangles
* Complex objects (combinations of simple shapes), like pedestrians, animals, and  cars
* The traffic sign as a whole (a combination of complex objects)

To achieve my goal of 95% validation accuracy, I focused on parameter tuning and created a rough table of key parameters to test.  For example, Normalize vs Min-Max Scaling, no Dropout vs Dropout (with different keep probabilities), batch size, training epochs, and learning rate.  One notable improvement was the removal of the dropouts in the convolutional layers.  After removing those dropouts, I increased the validation accuracy by 2-3%.  According to the original paper that proposed dropout layers, [Hinton, 2012](https://arxiv.org/pdf/1207.0580.pdf), dropout is more advantageous on fully (dense) connected layers because they contain more parameters than convolutional layers.  Higher parameter counts tend to have excessive co-adaptation in the neurons, which cause overfitting.  This plan adjusts and tests a little bit of everything including preprocessing, regularization, and model training to build a successful and accurate model.  




???
How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
???


![alt text][image4] ![alt text][image5]


-Model evaluation:
-Evaluate how well the loss and accuracy of the model for a given data set
-loss/cost?


One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.



## Test a Model on New Images

### Acquiring New Images

When I first conducted a Google search for German traffic signs, it was difficult to find a decent variety of interesting traffic signs on the web.  So, I decided to use Google translate to search for the traffic signs in German instead of English.  As a result, the new search results were significantly better in quality and quantity.  In addition, the images are unique, interesting, and challenging.  Finding good data is always half the battle!  Here are six German traffic signs that I found on the web with my expanded German vocabulary:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]

The first image might be difficult to classify because the snowflake is blurry.  Plus, there is snow sticking to the top of the sign.

The second image might be difficult to classify because the first digit (8) is confused with other similar digits with curves (2,3,5,6,9).

The third image might be difficult to classify because there is a busy background.

The fourth image might be difficult to classify because there are multiple signs in the image.

The fifth image might be difficult to classify because there are colorful autumn leaves in the background that could blend into the sign.

The last image might be difficult to classify because there is a slight offset angle of the sign with respect to the camera that could distort the true shape of the sign.





### Performance on New Images

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

### Model Certainty - Softmax Probabilities

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


