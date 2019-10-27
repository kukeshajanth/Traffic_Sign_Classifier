## Project: Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
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

[image1]: ./visual/color.png "Visualization 1"
[image2]: ./visual/traffic.png "Visualization 2"
[image8]: ./visual/gray.png "Visualization 3"
[image3]: ./final_test/ahead_only.png "Traffic Sign 1"
[image4]: ./final_test/Do-Not-Enter.png "Traffic Sign 2"
[image5]: ./final_test/keep_right.png "Traffic Sign 3"
[image6]: ./final_test/Road_Work.png "Traffic Sign 4"
[image7]: ./final_test/windling_road.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.
---

### Data Set Summary & Exploration

#### 1. After importing the dataset files I mostly used the methods len() and shape() to calculate number of examples and the shape of the images. I also imported the names of the labels for the traffic signs using the CSV library.  	

* Number of training examples = len(X_train) = 34799
* Number of validation examples = len(X_valid) = 4410
* Number of testing examples = len(X_test) = 12630
* Image data shape = X_train[0].shape = (32, 32, 3)
* Number of classes = len(all_labels) = 43

#### 2. Include an exploratory visualization of the dataset.

For the exploratory visualization of the data set. I have printed random images from the dataset with their labels.

![alt text][image1]

Then, I printed a bar chart showing how many examples of each class were available on the training set.

![alt text][image2]

I have converted the images from dataset to grayscale and normalize them.

![alt text][image8]



### Design and Test a Model Architecture

As a first step, I augmented the dataset using random rotation, random noise, intensity, blur. Finally, I concatenated the x_train dataset with these augmented images, thereby increasing the dataset size to 173995. 

```python
def random_noise(x):
    
    for i in range(len(x)):
        x[i] = sk.util.random_noise(x[i])
    
    return x

def intensity(x):
    
    for i in range(len(x)):
        v_min, v_max = np.percentile(x[i],(0.2,99.8))
        x[i] = exposure.rescale_intensity(x[i],in_range = (v_min,v_max))
    
    return x

def blur(x):
    
    for i in range(len(x)):
        
        x[i] =ndimage.uniform_filter(x[i], size=(11, 11, 1))
        
        
        return x
    
def random_rotation(x):
    
    for i in range(len(x)):
        
        random_degree = random.uniform(-25, 25)
        x[i] = sk.transform.rotate(x[i], random_degree)
        
        return x
    
```


Then, I decided to shuffle the examples using shuffle from the sklearn.utils library.

```python
aug_xtrain, aug_ytrain = shuffle(aug_xtrain, aug_ytrain)
```

After augmentation, my accuracy has improved from 90% to 95%.

For my final model I used the LeNet example from Lesson 8 as reference model. 

I have increase the depth of filters to extract more features and have used dropout layers to avoid overfitting.



| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Gray image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x24  |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x192 |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x192   |
| Convolution 3  	    | 1x1 stride, VALID padding, output = 1x1x768   |
| RELU					|												|
| Flatten				| output = 768									|
| Fully connected		| input = 768, output = 384       	            |
| RELU					|												|
| Dropout			    |												|
| Fully connected		| input = 384, output = 192       	            |
| RELU					|												|
| Dropout				|												|
| Fully connected		| input = 192, output = 43       	            |


To train the model I used 60 epochs, a batch size of 256 and a learning rate of 0.0007. Also, I have kept mean = 0 and stddev = 0.05.

For my training optimizers I used softmax_cross_entropy_with_logits to get a tensor representing the mean loss value to which I applied tf.reduce_mean to compute the mean of elements across dimensions of the result. Finally I applied minimize to the AdamOptimizer of the previous result.

My final model Validation Accuracy was 0.956 and the accuracy on the test image set is 0.94.

### Test a Model on New Images

I choose five German traffic signs found on the web to test my network.

Here are 5 of the  German traffic signs that I found on the web:

![alt text][image3] ![alt text][image4]


![alt text][image5] ![alt text][image6]

![alt text][image7] 


To make sure that

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| No entry       		| No entry   									|
| Slipery road     		| Slipery road 									|	
| Keep right			| Keep right									|
| Ahead only			| Ahead only									|	
| Road work      		| Road work     		    					|		


The model was able to correctly predict 5 other 5 traffic signs, which gives an accuracy of 100%.

Based on the comparison with the accuracy of the testing sample (0.958) , it has performed better on german images but fail to perform on different sets of images. Accuracy of the model can be improved by increasing the training data that is given to the model.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

I have initially converted the german images to gray scale before feeding through the network for prediction.

I have predicted the output using:

```python
with tf.Session() as sess:
    saver.restore(sess, './traffic')
    results = sess.run(tf.argmax(logits, 1), feed_dict = {x : xgray, drop : 1.0})
    results = [labels[n] for n in results]
    
    print(results)
    
```


For each of the new images, I have printed the model's softmax probabilities to show the certainty of the model's predictions (I have limited the output to the top 5 probabilities for each image). 

Do-Not-Enter.png:
No entry: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

windling_road.png:
Slippery road: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

keep_right.png:
Keep right: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

ahead_only.png:
Ahead only: 100.00%
Speed limit (20km/h): 0.00%
Speed limit (30km/h): 0.00%
Speed limit (50km/h): 0.00%
Speed limit (60km/h): 0.00%

Road_Work.png:
Road work: 100.00%
Bicycles crossing: 0.00%
Dangerous curve to the right: 0.00%
Bumpy road: 0.00%
No passing for vehicles over 3.5 metric tons: 0.00%




## Conclusion:


1 - I feel that with better augmentation accuracy of the model can be increased. I tried to perform augmentation using tensorflow , but I was not successful in doing so. I will try to implement this in future.

2 - Model performs poorly in images, where the training images of those particular signs are less. Also, I would like to experiment with different architecture or use transfer learning to improve the model.
