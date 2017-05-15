# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/hiromichinomata/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The image below is one of the samples.

![Sample image](./sample.png)

### Design and Test a Model Architecture

#### 1. How I preprocessed the image data.

As a first step, I decided to convert the image data type to float32 beacause I want to change the mean of image to zero.

Originally, I used numpy to subtract the mean from each image.
However by using tf.image.per_image_standardization, the better result was achieved for the images on the web because standard deviation is considered.

Also, the random change of  rotation, brightness, contrast are performed for better generalization.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
| :---------------------:	|   :---------------------------------------------------------:	|
| Input         		| 32x32x3 RGB image   					        |
| Convolution 5x5  	| 1x1 stride, valid padding, outputs 28x28x6	|
| ELU			|										|
| Max pooling	      	| 2x2 stride, outputs, 14x14x6				|
| Convolution 5x5	| 1x1 stride, valid padding, outputs 10x10x16	|
| ELU			| 		     								|
| Max pooling		| 2x2 stride, outputs, 5x5x16 				|
| Flatten			| Outputs 400 								|
| Fully connected	| Outputs 300								|
| Dropout			| 0.75 keep_prob							|
| ELU			| 										|
| Fully connected	| Outputs 200								|
| Dropout			| 0.75 keep_prob							|
| ELU			| 										|
| Fully connected	| Outputs 84								|
| Dropout			| 0.75 keep_prob							|
| ELU			| 										|
| Fully connected	| Outputs 43								|
| (Softmax)		| 		-						|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer.
At first, I fixed all hyperparameters except for batch size with small epoch.
After many attempts for hyperparamter search, batch size was fixed.
Then, I used the same one freedom way for rate and keep_prob and model structure little by little.

After some improvement, batch size was increased and tried again about hyperparameter search.
Then I fixed the epoch which achieved target accuracy.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.936
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

  LeNet

* What were some problems with the initial architecture?

  Accuracy was low compared with target.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  Dropout addition, change activation function from relu to elu, Fully connected layer addition, normalization way change were conducted.

* Which parameters were tuned? How were they adjusted and why?

  Batch size, leraning rate, keep probability of dropout, epochs are tuned.
  Without all parameters tuning, the target accuracy cannot be achieved for test set (0.93).

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

   Dopout addition significantly imroved the accuracy.
   On the other hand, activation funcion change didn't affect so much.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![sample](/images/0.jpg)
![sample](/images/1.jpg)
![sample](/images/2.jpg)
![sample](/images/3.jpg)
![sample](/images/4.jpg)

The second image might be difficult to classify because it contains multiple signs.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			            | Prediction	        		                			|
|:---------------------:|:---------------------------------------------:|
| Stop Sign         		| Priority road   							             		|
| Speed limit(30km/h)		| Priority road         							        	|
| Children crossing			| Children crossing       					    				|
| Speed Limit(50km/h)		| Speed Limit(50km/h)      					        		|
| Yield			            | No passing for vehicles over 3.5 metric tons 	|


The model was able to correctly guess 2 of the 5 traffic signs.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 288th cell of the Ipython notebook.

Here is the result.

Image 0: Stop
  * Priority road
    1.0
  * No passing
    2.11684e-10
  * No passing for vehicles over 3.5 metric tons
    3.65495e-13
  * Yield
    1.20289e-15
  * End of no passing
    5.16922e-18


Image 1: Speed limit (30km/h)
  * Priority road
    0.999943
  * Traffic signals
    3.63868e-05
  * Ahead only
    1.16267e-05
  * Yield
    6.31524e-06
  * Road work
    1.91603e-06


Image 2: Children crossing
  * Children crossing
    0.841416
  * Speed limit (20km/h)
    0.153635
  * Road narrows on the right
    0.00184727
  * Slippery road
    0.00142969
  * Bicycles crossing
    0.00127747


Image 3: Speed limit (50km/h)
  * Speed limit (50km/h)
    0.999943
  * Speed limit (80km/h)
    3.69199e-05
  * Speed limit (30km/h)
    2.05839e-05
  * Speed limit (60km/h)
    5.93463e-09
  * Speed limit (100km/h)
    1.44111e-13


Image 4: Yield
  * No passing for vehicles over 3.5 metric tons
    0.549298
  * Priority road
    0.403583
  * Stop
    0.0209008
  * No passing
    0.0161593
  * No vehicles
    0.00990566

Regarding image 1, original my estimation was the prediction is uncertain. However, neural net made a mistake with full confidence.
Based on the dataset, neural network might assume the region of interest is only one area.

On the other hand, regarding image 2, the prediction is a bit uncertain between Children crossing and Speed limit (20km/h). It is natural because the both signs have the red bounding with text in the center.
