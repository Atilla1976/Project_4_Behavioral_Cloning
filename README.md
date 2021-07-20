# **Behavioral Cloning** 

---


In this project, a convolutional neural network (CNN) is trained to clone a human driving behavior.

For data collection a simulator provides by Udacity could be used. The simulator collects images and the associate steering angles.
The collected dataset is used to train the convolutional neural network. Subsequently this model is used to drive the vehicle autonomously around the track in the simulator.


The goals / steps of this project are the following:
* Using the simulator to collect data of good driving behavior
* Building, a convolution neural network in Keras that predicts steering angles from images
* Training and validating the model with a training and validation set
* Testing that the model successfully drives around track one without leaving the road
* Summary of the results


[//]: # (Image References)

[image1]: ./P4_images/CNN.png "NVIDIA CNN"
[image2]: ./P4_images/layers.png "Layers"
[image3]: ./P4_images/flipped_image.png "Normal Image"
[image4]: ./P4_images/different_perspectives.png "Flipped Image"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---




### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/Atilla1976/SDCarND_P4_Behavioral_Cloning/blob/main/model.py) containing the script to create and train the model
* [drive.py](https://github.com/Atilla1976/SDCarND_P4_Behavioral_Cloning/blob/main/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/Atilla1976/SDCarND_P4_Behavioral_Cloning/blob/main/model.h5) containing a trained convolution neural network 
* Readme.md summarizing the results
* [run1.mp4](https://github.com/Atilla1976/SDCarND_P4_Behavioral_Cloning/blob/main/run1.mp4) - video recording of my vehicle driving autonomously around the track for at least one full lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I did not build the neural network from scratch. I used the NVIDIA convolutional neural network which was developed exactly for this purpose. This is structured as follows:

![alt text][image1]


The convolutional layers were chosen by NVIDIA empirically through a series of experiments that varied layer configurations. Strided convolutionals are used in the first three convolutional layers with a 2x2 stride and a 5x5 kernel and a non-strided convolution with a 3x3 kernel size in the last two convolutional layers.


The model includes RELU layers to introduce nonlinearity (code line 89, 92 and 94), and the data is normalized in the model using a Keras lambda layer (code line 86). 

#### 2. Attempts to reduce overfitting in the model

The model contains four dropout layers in order to reduce overfitting (model.py lines 91, 100, 102 and 104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 109).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only the dataset provided by Udacity. 
For details about why I only used this data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Final Model Architecture

The final model architecture (model.py lines 85-106) consists of the NVIDIA convolution neural network as shown above. I implemented 4 dropout layers each with a rate of 0.25 to avoid overfitting.



```sh
model.summary()
```
returns the following information about the shape and sequence of the all network layers

![alt text][image2]

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, my approach was initially to record the following driving maneuvers:
* two laps on track one using center lane driving.
* driving the track in opposite direction to avoid a "left-drift" due driving counterclockwise
* recording recovery driving from the side of the road back to the center

But then I decided  to use only the provided dataset.

* The effect of an assumed drift to the left could just as easily be prevented in the data augmentation stage by flipping the dataset.

![alt text][image3]


* And even recording recovery driving can be replaced by using the multicamera dataset. Here are three different datasets from three camera positions - one in the center, one on the left and one on the right. From the perspective of the left camera, the steering angle is less than the steering angle from the center camera. From the right camera's perspective, the steering angle would be larger than the angle from the center camera. During training I fed the left and right camera images to my model as if they were coming from the center camera. For this reason, I initially assumed a correction value for the steering angle of +/- 0.2 for the left and right camera images. This value proved to be expedient.


![alt text][image4]



I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3. I used an adam optimizer so that manually training the learning rate wasn't necessary.
