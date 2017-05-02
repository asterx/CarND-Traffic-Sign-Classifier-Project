**Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[train_images]: ./out/train_images.png "Original train images"
[distribution]: ./out/distribution.png "Distribution"
[images_generated]: ./out/test_images_generated.png "Generated images for better training"
[net]: ./out/net.png "Net"
[test_images_web]: ./out/test_images_web.png "Test images from web"
[test_images]: ./out/test_images.png "Test images from web - results"
[learing_progress_accuracy]: ./out/learing_progress_accuracy.png "Learning progress accuracy"
[learing_progress_loss]: ./out/learing_progress_loss.png "Learning progress loss"
[probabilities]: ./out/probabilities.png "Probabilities"
[feature_map]: ./out/feature_map_conv_relu_1.png "Feature Map"

## Rubric Points
###Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Train images in original dataset][train_images]

![Class distribution][distribution]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of 
images in the set, number of images for each class, etc.)

Firstly I normalized data (in case of Gradient Descent normalization gives the error surface a more spherical shape and as a result it works much better). Then applied one-hot encoding to convert label numbers to vectors. After some experiments it become obvious that size and quality of train data set matters. Original images were pre-processed (brightness, zoom and etc. I also tried gray scale but it didn't helped a lot so I removed it). Then I generated some images using different modifications (angle, brightness, blur and etc.).

![Generated images][images_generated]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model looks like this:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 2x2    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 14x14x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 7x7x64 				|
| Fully connected		| 1024x1024       									|
| RELU					|												|
| DROPOUT					|	50%											|
| Fully connected		| 1024x1024       									|
| RELU					|												|
| DROPOUT					|	50%											|
| OUT					|												|

![Net][net]


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
Params for trainging final version of model:
* OPTIMIZER = AdamOptimizer
* EPOCHS = 12
* BATCH_SIZE = 128
* BASE_RATE = 0.001

I used AdamOptimizer for many reasons, but most important one is that it uses moving averages of the parameters. This enables AdamOptimizer to use larger effective step size and the algorithm will converge to this step size without fine tuning. Of course there is always a down side. In this case it is that AdamOptimizer requires more computation to be performed for each parameter in each training step (but fortunately we have aws cloud and given task is not that big). 
Number of epochs, batch size and base rate were chosen after many experiments.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started my research with well-known architecture called LeNet (It's pretty easy to implement and training model is quite fast). Firstly I used only images from training data set and didn't generated any additional data. But after several experiments I realized that I can't achieve expected accuracy. Adding generated images to training dataset makes life much easier.
Then I tried various combinations of conv and fully-connected layers. Tried to change depth for conv layers and number of neurons for fully-connected ones. Also I tried to put dropout in different places. After many hours of experiments I decided that model described above (see table and image) is good enough for given task.

![Learing progress accuracy][learing_progress_accuracy]
![Learing progress loss][learing_progress_loss]

Final hyperparameters:
* OPTIMIZER = AdamOptimizer
* MEAN = 0.0
* STDDEV = 0.01
* TRAIN_SIZE = 0.8
* EPOCHS = 12
* BATCH_SIZE = 128
* BASE_RATE = 0.001

My final model results were:
* training set accuracy: 98%
* validation set accuracy of 95%
* test set accuracy of 88%
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web (images after post-processing):
![Test images from web][test_images_web]

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)      		| Speed limit (50km/h)   									| 
| Speed limit (60km/h)     			| Speed limit (60km/h) 										|
| Pedestrians					| Road narrows on the right											|
| No passing	      		| No passing					 				|
| Children crossing			| Children crossing      							|
| Yield			| Yield      							|
| Stop			| Stop      							|
| Road work			| Road work      							|

![Test images from web with results][test_images]

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 88%.
The only sign that caused problems was "Pedestrians". I think it happen b/c of several reasons.
* Some images are too dark (which means that more pre-procesing is required; it might be helpful to remove too dark images from training set)
* Pedestrians and Road narrows on the right look similar in some ways (same shape of the sign, same dark content). I think including more different images of these signs in testing set could help network learn difference better.

We should also keep in mind fact that success classification result 88% is significantly less then 95% and 95% on the test and validation dataset respectively. It is clear that model overfitting and there is more work to be done here. I see at least two obvious steps that could help solve this issue - spend more time with configuring right regularization and better prepared training data. I'll definetly pay more attention to this in my next assignment.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![Probabilities][probabilities]

First couple of signs are quite easy and models is very certain about it's predictions (specially with 1st one - "Speed limit (50km/h)"). Signs share similar round shape, red color and digits inside. 100% certainty for 50mk/h is a bad sign of over fitting, I'd expect probability graph to look more like one for 60km/h (which indicates that model understood shape and colors and choosing between digits). 

3d sign ("Pedestrians") shows that training set have issues and doesn't contain enough images with that and other signs.
Model is certain about shape and color sometimes but final result is absolutely incorrect.

4th sign ("No passing") is again a good example that shows that model knows about shape and color and making deciding based on more higher level.

Rest of results again shows signs of over fitting.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
![Feature Map][feature_map]

This finally step really can help making network better. It shows weak spots, for example 2nd and 4th images shows that I still can improve my net and training data.
