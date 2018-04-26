# **Behavioral Cloning** 

## Using Deep Learning to Clone Driving Behavior

### Goals
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

![alt text][image01]
 

The primary goal of my project is to teach a Convolutional Neural Network (CNN) to drive a car in a simulator provided by Udacity, arcade-style. The vehicle is equipped with 3 front sensors- 1 center camera and 2 side cameras. Here is a set of example images from the car’s point of view at one instant in time on the training track:

 
![alt text][image02]

Left Camera

![alt text][image03]

Center Camera

![alt text][image04]

Right Camera

 
The simulator has two modes: training and autonomous. For training mode, the sensors output a video stream and records the values of steering angle, speed, throttle, and brake. Due to the many interesting features of the track (sharp turns, road textures, road borders, etc.), it is crucial to collect good training data to ensure a successful model for this project. For autonomous mode, the end to end deep learning model processes image data from its sensors and makes a single prediction for the steering angle. This actually turns out to be a regression network instead of a classification network, since the output layer of the model outputs a single node (steering angle). 


### Training Data Strategy

My strategy for collecting training data focused on the three following areas: normal laps, recovery lap, and generalization laps.  I started collecting the training data by carefully driving the car as close to the middle of the road as possible even when making turns.  I completed 3 normal laps around track one, which is the test track to drive the car autonomously.  Next I focused on recovery training to teach the car what to do when it’s off to the side of the road.  This was also performed on track one.  I only collected recovery data when the car is driving from the side of the road back toward the middle.  Each recovery interval was short lasting about 2-3 seconds.  I trained for approximately one lap alternating the recoveries from the left and right sides back to the middle.  To prevent the CNN from memorizing track one, I collected data by driving the car in the opposite direction on track one to help the model generalize.  Driving in the opposite direction essentially gives the model a brand new track to learn, which helps the model generalize better.  Furthermore, I collected some training data on the second track, which is more challenging with a mix of steep hills, sharp turns, and road debris.  I trained on flat terrain with greater emphasis on smoothly executing turns on track two.  Overall, my final training dataset consisted of 39,000 images.  Below is the distribution of the steering angles that I recorded:               

![alt text][image05]
 

Histogram of Steering Angles: the range [-1, 1] corresponds to the angle range -/+ 25 deg'

Due to the characteristics of the training track, there are more straight sections on the track than curve sections and more left turns than right turns causing an unbalanced dataset.  To combat the angle bias, let’s try an effective technique called data augmentation.

 

The left/right skew is due to driving the car around the track in one direction only and can be eliminated by flipping each recorded image and its corresponding steering angle. More troublesome is the bias to driving straight: the rare cases, when a large steering angle recorded are also the most important ones if the car is to stay on the road.


### Data Augmentation

Flipping Images And Steering Measurements

A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement.




-To train the network, I launched an AWS EC2 instance attached on a GPU.

- Used Keras framework with TensorFlow backend... 

I built/developed a behavioral cloning network. I drove a car around a flat-terrain track (video game style) using a simulator provided by Udacity, collected data, then trained a deep neural network to do the driving for me.

 
The model had to learn how to handle sharp turns, varying road textures, and different borders that lined the edges of the road.

 

Archive:

As a starting point, I collected the training data by carefully driving the car as close to the middle of the road as possible even when making turns for one lap around the track. After one complete lap, I noticed that a potential steering angle bias was developing due to the characteristics of the training track. Overall, there are more straight sections on the track than curve sections (and more left turns than right turns).  As a result, the steering angle was very close to zero most of the time.  I collected additional training data on two laps of center lane driving with greater emphasis on curve training than straight driving training to help balance the dataset.  Next I focused on recovery training.      



My project includes the following files:
* model.py containing the script to create and train the model using a Convolutional Neural Network (CNN)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained CNN
* writeup_report.md summarizing the results
* video.mp4 containing a video recording of my vehicle driving autonomously one lap around the track

<i>You can find my project code [here](https://github.com/bkaewell/self-driving-car/blob/master/P3-behavioral-cloning/)</i>

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```







[//]: # (Image References)
[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/recovery_bridge_before.jpg "Recovery Image"
[image4]: ./examples/recovery_bridge_after.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[image01]: ./examples/birds_eye.png "Bird's Eye"
[image02]: ./examples/left_2018_04_20_19_13_00_008.jpg "Left Camera Sensor"
[image03]: ./examples/center_2018_04_20_19_13_00_008.jpg "Center Camera Sensor"
[image04]: ./examples/right_2018_04_20_19_13_00_008.jpg "Right Camera Sensor"
[image05]: ./examples/steering_angles_histogram.jpg "Steering Angles Histogram"

---


## Model Architecture and Training Strategy

-1. An appropriate model architecture has been employed
My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 


<i>The model I used was the [NVIDIA model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)</i>

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

-2. Attempts to reduce overfitting in the model
The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

-3. Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

-4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Documentation

-1. Solution Design Approach
The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

-2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

-3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
