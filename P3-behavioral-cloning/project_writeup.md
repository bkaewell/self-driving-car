# **Behavioral Cloning**

## Using Deep Learning to Clone Driving Behavior

### Goals
* Use the simulator provided by Udacity to collect data of good driving behavior
* Build a Convolutional Neural Network (CNN) in Keras with TensorFlow back-end that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---

![alt text][image01]
Example of Center Lane Driving



The primary goal of my project is to teach a CNN to drive a car arcade-style in a simulator. The simulator has two modes: training and autonomous. For training mode, the sensors output a video stream and records the values of steering angle, speed, throttle, and brake. Due to the many interesting features of the track (sharp turns, road textures, road borders, etc.), it is crucial to collect solid training data to ensure a successful model for this project. For autonomous mode, the end to end deep learning model processes image data from its sensors and makes a single prediction for the steering angle. This actually turns out to be a regression network instead of a classification network, since the output layer of the model outputs a single node (steering angle).  The vehicle is equipped with 3 front-facing sensors located in the center and on both sides. Here is a set of example images from the car’s point of view at one instant in time on the training track:

![alt text][image02]

Left Camera

![alt text][image03]

Center Camera

![alt text][image04]

Right Camera


### Training Data Strategy

My strategy for collecting training data focused on the following areas: normal laps, recovery laps, and generalization laps. Although image data was available from 3 different cameras, I decided to only use images recorded by the center camera.

I first recorded the training data by carefully driving the car as close to the middle of the road as possible even when making turns.  To capture good driving behavior, I recorded 3 normal laps on track one using center lane driving.  This is the designated track to evaluate the model’s performance for an autonomous vehicle.  Here is an example image of center lane driving:???

I then recorded the vehicle recovering from the left side and right side of the road back to the center so that the vehicle would learn to return to the middle when it wanders off to the side.  This was also performed on track one.  It was very important to disable recording when I intentionally drove the car to either side of the road to setup the recovery training because I did not want to teach the network bad driving habits to drift off to the side.  So I only collected recovery data when the car was driving from the side of the road back toward the middle.  Each recovery training interval was very short, lasting about 2-3 seconds (~50 images).  I trained for approximately one lap alternating between sides to create a balanced and diverse dataset.  One lap was sufficient because it allowed the network to learn different road textures and borders for side recoveries.  These images show what a recovery looks like starting from the right side on the bridge: 


![alt text][image06]

Recovery Start (Right Shoulder)

![alt text][image07]

Recovery End (Re-centered)


To prevent the CNN from memorizing track one, I recorded the car driving in the opposite direction to collect more data points.  Driving in the opposite direction essentially gives the model a brand new track to learn, which helps the model generalize better.  Furthermore, I collected some training data on track two, which is more challenging with a mix of steep hills, sharp turns, and road debris.  I only trained on flat terrain with greater emphasis on smoothly executing sharp turns.

Overall, my final training dataset consisted of 16,430 images.  Below is the distribution of the steering angles that I recorded:


![alt text][image05]

As expected, the steering angles were virtually zero most of the time due to the loop in the training track that had more straight sections than curve sections.  Aside from the very strong peak in the middle of the distribution, I am pleasantly surprised by how well the steering angles were balanced for the left turns (negative values) and right turns (positive values) in my training dataset.  However, right turns were still a little under sampled compared to left turns.     
    
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that ...??????????????????

With the combination of training on normal laps, recovery laps, and generalization laps, I discovered that it was not enough data to train my model to drive properly.  After running my newly trained model and predicting steering measurements in autonomous mode, my car consistently crashed into the lake on the sharp right turn after the bridge.  Things didn’t go perfectly, so now it’s time to discuss data augmentation to combat the biases.

The final step was to run the simulator to see how well the car was driving around track one.  There was one specific spot where the vehicle fell off the track on the first sharp right turn after the bridge.  To improve the driving behavior in this case, I decided to augment the data for the extreme positive steering angles.  At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.


### Image and Data Augmentation

To augment the dataset, I also flipped images and angles thinking that this would eliminate the left/right data skew.  For example, here is an image that has then been flipped:

![alt text][image08] ![alt text][image09]

Left Turn Original

![alt text][image09]

Flipped


In addition, I added images above a certain angle threshold (-/+ 0.1) and randomly varied the brightness of the image thinking that this would help reinforce both left and right turns while teaching the model new brightness patterns.




![alt text][image10]

Sharp Right Turn Original

![alt text][image11]

Sharp Right Turn Sunny

![alt text][image12]

Sharp Right Turn Shady




Flipping Images And Steering Measurements

A effective technique for helping with the left turn bias involves flipping images and taking the opposite sign of the steering measurement.



### Model Architecture and Training Documentation


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

-2. Final Model Architecture

The final model architecture (model.py lines 67-92) consisted of a CNN with the following layers and layer sizes ...

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

To augment the data set, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary. 

    




### Model Architecture and Training

I implemented a behavioral cloning network using Keras framework with Tensorflow back-end.

<i>My model architecture was based on NVIDIA's end to end deep learning for self driving cars found [here](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)</i>

My model consists of a CNN with strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers.  It had depths between 24 and 64 (model.py lines 76-80)

The model includes RELU layers to introduce nonlinearity (model.py lines 76-80), and the data is normalized in the model using a Keras lambda layer (model.py line 70).  It also uses a cropping layer to remove the horizon from the top of the image and the hood of the car from the bottom (model.py line 73). 

The model contains a dropout regularization layer at the point in the network with the most parameters in order to reduce overfitting (model.py line 86). Here is a visualization of the CNN architecture:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
= ==================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 67, 320, 3)    0           lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 32, 158, 24)   1824        cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
= ==================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
```

The model was trained and validated on different datasets to ensure that the model was not overfitting (model.py lines 67-92). The model was tested by running it through the simulator and verifying that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 91).

I trained the network by launching an AWS EC2 instance attached on a GPU.  For any given set of hyperparameters, the loss typically stopped decreasing around 5 epochs.


### Results

The model had to learn how to handle sharp turns, varying road textures, and different borders that lined the edges of the road.

<i>You can find a video of my car navigating the test track in autonomous mode [here](https://youtu.be/a6wvZnbKRT4)</i>


### References

<i>You can find my project code [here](https://github.com/bkaewell/self-driving-car/blob/master/P3-behavioral-cloning/)</i>

My project includes the following files:
* model.py containing the script to create and train the model using a Convolutional Neural Network (CNN)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained CNN
* writeup_report.md summarizing the results
* video.mp4 containing a video recording of my vehicle driving autonomously one lap around the track

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

[//]: # (Image References)
[image01]: ./examples/birds_eye.png "Bird's Eye"
[image02]: ./examples/left_2018_04_20_19_13_00_008.jpg "Left Camera Sensor"
[image03]: ./examples/center_2018_04_20_19_13_00_008.jpg "Center Camera Sensor"
[image04]: ./examples/right_2018_04_20_19_13_00_008.jpg "Right Camera Sensor"
[image05]: ./examples/steering_angles_histogram.jpg "Steering Angles Histogram"
[image06]: ./examples/recovery_bridge_before.jpg "Recovery Image Before"
[image07]: ./examples/recovery_bridge_after.jpg "Recovery Image After"
[image08]: ./examples/left_turn_original.jpg "Left Turn Original Data Aug"
[image09c]: ./examples/left_turn_flipped.jpg "Left Turn Flippled Data Aug"
[image10]: ./examples/sharp_right_turn_original.jpg "Right Turn Original Aug"
[image11b]: ./examples/sharp_right_turn_sunny_aug.jpg "Right Turn Brightened Aug"
[image12b]: ./examples/sharp_right_turn_shady_aug.jpg "Right Turn Darkened Aug"


