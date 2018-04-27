import csv
import cv2
import numpy as np
import random

# Helper function to randomize image brightness
def modify_brightness(image):
    # Convert to HLS
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_hls = np.array(image_hls, dtype = np.float64)

    # Generate value between 0.5 and 1.5
    value = np.random.uniform() + 0.5

    # Scale pixel values for channel 1 (lightness)
    image_hls[:,:,1] = image_hls[:,:,1] * value

    # Clip all values above 255 to 255
    image_hls[:,:,1][image_hls[:,:,1] > 255]= 255
    image_hls = np.array(image_hls, dtype = np.uint8)
    image_rgb = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)

    return image_rgb


# Read csv file:
lines = []
with open('driving_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# Read images and extract steering angle measurements from csv file:
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'driving_data/IMG/' + filename
    image = cv2.imread(current_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Augment training data:
aug_images = []
aug_measurements = []
for image,measurement in zip(images, measurements):  
    aug_images.append(image)
    aug_measurements.append(measurement)

    # Flip image (horizontally) and invert angle
    aug_images.append(cv2.flip(image,1))
    aug_measurements.append(measurement*-1.0)

    # Duplicate steering angles greater than 2.5 degrees 
    # And randomize image brightness  
    # The range [-1, 1] corresponds to steering angle range of +/- 25 degrees
    if (abs(measurement) > 0.1):
        image_out = modify_brightness(image)
        aug_images.append(image_out)
        aug_measurements.append(measurement)
       
    # Duplicate just right turn data exceeding 5 degrees (positive steering angles)
    # And randomize image brightness:
    if (measurement > 0.2):
        image_out = modify_brightness(image)
        aug_images.append(image_out)
        aug_measurements.append(measurement)
        

X_train = np.array(aug_images)
y_train = np.array(aug_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

# NVIDIA End to End Deep Learning Model (image to steering angle):
model = Sequential()

# 2 step Preprocessing: Normalizing, mean-centering 
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# Crop out horizon from the top and the car hood from the bottom
model.add(Cropping2D(cropping=((70,23),(0,0))))

# Convolutional layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Flatten())

# Fully Connected Layers:
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model_nvda.h5')
model.summary()