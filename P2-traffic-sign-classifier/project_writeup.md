# **Traffic Sign Classification with Deep Learning**

---

<i>You can find my project code [here](https://github.com/bkaewell/self-driving-car/blob/master/P2-traffic-sign-classifier/Traffic_Sign_Classifier.ipynb)</i>


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
[image4]: ./output/traffic_sign_model_accuracy2.png "Model Accuracy"
[image5]: ./output/traffic_sign_cross_entropy2.png "Cross Entropy"
[image6]: ./output/traffic_signs_resized_web.jpg "Traffic Sign Test 1-6"


---


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

After exploring a limited subset of the training data, I decided that the traffic sign colors would be relevant so I did not convert to grayscale.  It just seems like blobs of color especially for a red stop sign could probably aid the learning experience of a neural network.  As a first step, I shuffled the data to reduce the variance in the originally sorted training set and prevent overfitting.  Then I decided to normalize the image data using the min-max scaling technique because I wanted to build a well-conditioned model with zero mean and small variance for the RGB pixels to keep the model very uncertain about things.  Furthermore, it makes it easier for the optimizer to proceed numerically (faster searches to reach a solution).  

Here is an example of a traffic sign image before and after min-max scaling normalization:


![alt text][image3]


The difference between the original data set and the preprocessed data set is the dynamic range of the pixels due to normalization with min-max scaling to a pixel intensity range of (0.1, 0.9).  In the figure above, notice the whites are not as white and the darks are not as dark producing a small variance pixel to pixel.


### Model Architecture

My final model architecture is a multi-layer CNN to classify the traffic signs from Germany.  It consisted of the following layers resembling the "LeNet-5" function:

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

To train the model, I used an Adam Optimizer with the default parameter settings. After numerous trials, I used a batch size of 100 and 45 for the number of training epochs. I used a learning rate of 0.001. The dropout rate for the fully connected layers had a keep probability of 0.60 meaning 60% of the neurons were retained.  I am starting to feel that adjusting the number of epochs, learning rate, and dropout rate is a losing strategy.  If I had realized this sooner, I would have spent more time building multiple prediction algorithms instead and selected the one with the highest validation accuracy.  


### Results

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.945
* test set accuracy of 0.931


### Solution Approach

I used a CNN architecture called "LeNet-5" and implemented it in TensorFlow and Python. I decided this was suitable for the current problem because the CNN learns to recognize basic lines and curves, then shapes and color blobs, and then increasingly complex objects within the image.  It is a very powerful method to learn and classify images similar to how humans perceive images.  In this traffic sign case, the levels of hierarchy are:

* Lines and curves
* Blobs of color, simple shapes, like circles and triangles
* Complex objects (combinations of simple shapes), like pedestrians, animals, and  cars
* The traffic sign as a whole (a combination of complex objects)

To achieve my goal of 93% validation accuracy, I focused on preprocessing and model parameter tuning.  I created a rough table of key parameters to test and configure.  For example, Normalize vs Min-Max Scaling, no Dropout vs Dropout (with different keep probabilities), batch size, training epochs, and learning rate.  One notable improvement was the removal of the dropouts in the convolutional layers.  After removing those dropouts, I increased the validation accuracy by 3%.  According to the original paper that proposed dropout layers, Hinton, 2012, dropout is more advantageous on fully (dense) connected layers because they contain more parameters than convolutional layers. Higher parameter counts tend to have excessive and complex co-adaptation of feature detectors in the training data, which causes overfitting. This plan adjusts and tests a little bit of everything including preprocessing, regularization, and model training to build a successful and accurate model.

The measured accuracy results from training, validation, and testing verify that the final model is highly accurate.  The training accuracy just lets us know how well the model is training and testing on the same data set.  The measured value of 99.8% is almost identical to the standard 100% value, which guarantees correct classification of each training set example.  The final validation accuracy of 94.5% estimates how well my model has been trained.  The validation set is used to measure the ability of the model to generalize on unseen data.  Once I developed a fully-trained model, I evaluated it against the test set.  The final test accuracy of 93.1% estimates how well my model performed on a set of unseen real-world examples. 

To visualize the model training history, below is a plot showing the model accuracy on the training and validation data sets over training epochs.  The gap between the training and validation accuracy indicates the amount of overfitting.  In the plot, the validation accuracy tracks the training accuracy fairly well, but there is a slight gap between the two curves, which causes some overfitting. One way to reduce the amount of overfitting is to increase the model's capacity (i.e. increase the number of parameters).   

![alt text][image4]

Another useful thing to track during training is the loss since it is evaluated on the individual batches during the forward pass.  Based on the shape of the curve, we can determine the learning rate.  With lower learning rates, the shape is linear.  With higher learning rates, the shape is exponential.  Ideally, you want a "Goldilocks" learning rate, that is directly in the middle.  In the plot, the results from training show that the learning rate is pretty good.  Here is a plot of the loss or cross entropy over training epochs:

![alt text][image5]



## Test a Model on New Images

### Acquiring New Images

When I first conducted a Google search for German traffic signs, it was difficult to find a decent variety of interesting traffic signs on the web.  So, I decided to use Google translate to search for the traffic signs in German instead of English.  As a result, the new search results were significantly better in quality and quantity.  In addition, the images are unique, diverse, interesting, and challenging.  Finding good data is always half the battle!  Here are six German traffic signs that I found on the web with my newly expanded German vocabulary:


![alt text][image6]


The first image might be difficult to classify because the snowflake is blurry.  Plus, there is snow sticking to the top of the sign causing some obstruction.

The second image might be difficult to classify because it contains values.

The third image might be difficult to classify because there is a busy background.

The fourth image might be difficult to classify because there are multiple signs in the image.

The fifth image might be difficult to classify because there are colorful autumn leaves in the background that could blend into the sign.

The last image might be difficult to classify because there is a slight offset angle of the sign with respect to the camera that could distort the true shape of the sign.


### Performance on New Images

Here are the results of the prediction:

| Input Image                                   | Prediction                                   | 
|:---------------------------------------------:|:--------------------------------------------:| 
| Beware of ice/snow                            | Beware of ice/snow                           | 
| Speed limit (80km/h)                          | Speed limit (30km/h)                         |
| Stop                                          | Stop                                         |
| Road work                                     | Road work                                    |
| No passing for vehicles over 3.5 metric tons  | No passing for vehicles over 3.5 metric tons |
| Wild animals crossing                         | Wild animals crossing                        |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares pretty favorably to the accuracy on the test set of 93.1% considering I did my own cropping and resizing of random real-world images. I am not surprised that the model did not predict the correct value of the speed limit sign.   


### Model Certainty - Softmax Probabilities

The code for making predictions on my final model is located in cell #33 of the Ipython notebook.

For the first image, the model is relatively sure that this is a beware of ice/snow sign (probability of 0.84), and the image does contain a beware of ice/snow sign.  This result is fairly impressive because it doesn’t get any more real than snow sticking to a beware of ice/snow sign!


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .83         			| Beware of ice/snow   							| 
| .16     				| Children crossing 							|
| .00					| Slippery road									|
| .00	      			| Road narrows on the right					 	|
| .00				    | Bicycles crossing      						|



The second sign was not recognized correctly as the model predicted 30 km/h instead of 80 km/h.  Not only did it get the prediction wrong, but the next highest probability was end of 80 km/h with slashes through it!  This is very interesting since 80 km/h with slashes is more complex than 80 km/h without slashes.  As it turns out, based on the histogram of the training data set distribution, the speed limit 80 km/h is a majority class and the end of speed limit 80 km/h is a minority class.  In fact, the ratio between the two classes is 4:1!  The predicted class of speed limit 30 km/h had one of the most training examples.  This is a prime example where unbalanced distribution exposed the classifier’s flaws.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| Speed limit (30km/h)   						| 
| .04     				| End of speed limit (80km/h)					|
| .00					| Speed limit (80km/h)							|
| .00	      			| Speed limit (20km/h)					 		|
| .00				    | Speed limit (50km/h)      					|



The third sign was recognized correctly with 99% confidence.  Notice how the top 5 have either letters or number inside the sign.  No surprises here. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| .00     				| Speed limit (30km/h) 							|
| .00					| Speed limit (80km/h)							|
| .00	      			| Speed limit (60km/h)					 		|
| .00				    | Speed limit (50km/h)      					|



The fourth sign was recognized correctly with 100% confidence.  Top 5 contains objects inside the sign.  Again, no surpises. 


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00         			| Road work   									| 
| .00     				| Bicycles crossing 							|
| .00					| Keep right									|
| .00	      			| Bumpy Road					 				|
| .00				    | Dangerous curve to the right      			|



The fifth sign was recognized correctly with 100% confidence.  The classifier is firing on all cylinders here when the top two probabilities are “no passing” and “end of no passing” for vehicles over 3.5 metric tons.  Although the second best probability was miniscule, it’s still interesting that it was listed in the top 5.  It demonstrates a well behaving model for this particular real-world image.  Obviously in the real world the difference between a pass zone and no pass zone is critical and it’s important to get that right 100% of the time.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.00         			| No passing for vehicles over 3.5 metric tons  | 
| .00     				| End of no passing by vehicles over 3.5 metric |
| .00					| Priority road									|
| .00	      			| No passing					 				|
| .00				    | Dangerous curve to the right      			|



The sixth sign was recognized correctly with a 99% confidence.  I wonder if self-driving cars will use extra sensors and alert the passengers inside the car for wild animal zones since their crossings are totally unpredictable.


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Wild animals crossing   						| 
| .01     				| Bicycles crossing 							|
| .00					| Slippery road									|
| .00	      			| Road narrows on the right					 	|
| .00				    | Dangerous curve to the left      				|

