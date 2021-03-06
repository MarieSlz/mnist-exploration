## Exploration of the MNIST dataset

All the graphs and figures displayed in this document are reproducible using the main.py file in the src folder.

**Overall Comments**
- Test an train sets are the one predefined for the MNIST dataset. I am using this code to download the [dataset](https://github.com/tensorflow/tensorflow/blob/a5d8217c4ed90041bea2616c14a8ddcf11ec8c03/tensorflow/examples/tutorials/mnist/input_data.py).
- The training is limited to a single epoch to save computation time.

![Sample](../output/figs/sample_mnist_images.png)

## Part 1

**Choice of the Convolutional Neural Network**

I am using the LeNet network. It offers at the same time a relatively good accuracy and a fast training time. The goal of this project is to focus on the impact of noise rather than to select the best hyperparameters, hence I am using parameters found in the [literature](http://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/).

**What is the test error rate?**

The error rate is 1.41% after only one epoch of training on the original dataset.

**What is the test set error rate for each class? Are some classes more challenging than others to distinguish from each other? Why?**

We can use different metrics to report accuracy for multiclass classification. Below are reported:
- the precision rate: percentage of retrieved instances that are relevant tp/(tp+fp).
- the recall rate: percentage of relevant instances that are retrieved tp/(tp+fn).  

Abbreviations: tp stands for true positive, fp for false positive and fn for false negative

![Precision](../output/figs/output_0/precision.png)
![Recall](../output/figs/output_0/recall.png)

Accuracy being high for most of the classes, precision and recall are not the good metrics to report. Hence, I will report the test error rate per class which I will define as the percentage of retrieved instances that are non-relevant (1-tp/(tp+fp))
The test set error for each class is displayed below. We see on this graph that the error rates are not the same for all the classes.

![Test Error](../output/figs/output_0/test_error.png)

To have a good overview of the misclassification happening, the confusion matrix is the good accuracy measure to display.

![Confusion Matrix](../output/figs/output_0/confusion_matrix.png)

Analyzing the confusion matrix:
- we retrieve the metrics displayed on the Barplot above by looking at the columns (percentage of retrieved instances that are non-relevant).
- by looking at the rows of the matrix, we know which classes have a high percentage of relevant instances that are not retrieved.

Some classes are more difficult to distinguish from each other, 7s and 1s can be misclassified, this is also the case for 3s and 5s, 8s and 9s.

While a human would be able to differentiate a 7 and a 1 by looking at the size of the top bar, it's not that easy to reproduce this behavior with a neural network which takes as an input individual pixels.

**Based only on information gathered in the first epoch of training, do you think that the model would benefit from more training time? Why?**

The loss per batch during the first epoch of training is displayed below. The decrease of the loss is less important at the end of the epoch that it was at the beginning, but it seems to be still decreasing. 
To know for sure that we should continue the training, we would need to track the validation loss not the training loss. Only monitoring the loss on the training set could lead to overfitting.

![Loss](../output/figs/output_0/loss_first_epoch.png)

**Besides training for a longer time, what would you do to improve accuracy?**

Different methods could be used, however we have to keep in mind that the final goal is to improve generalization (and accuracy on the test set) and avoid overfitting.
- Algorithm tuning: Optimizing the weight initialization, the number of batches, the optimization function.
- Changing model: The network I chose is fairly simple, but using something like AlexNet would give better results. It's always possible to try to improve the model by adding layers, or changing activation functions.
- Work on the inputs: Get more data (even if here we already have a training sample of a good size), invent more data (by randomly modifying version of existing images)

## Part 2: Add Gaussian Noise to the Inputs

In this part, I added Gaussian noise to the training set images. The mean is always set to 0 and the standard deviation respectively to 8, 32 and 128.

The test error rate for the three situations is displayed below. Comparing the test error rate to the test error rate obtained in the first part, we notice that in the first two situations, it doesn't seem to affect the accuracy of the classifier.
Going further, the test error is even lower than in the first experiment (1.41% vs 1.32% and 1.35%). Of course to have a strong conclusion, one would need to repeat those experiments to be sure it's not a coincidence, but we can already make suppositions.

![Test Error](../output/figs/image_noise.png)

**What are the implications of the dependence of accuracy on noise if you were to deploy a production classifier? How much noise do you think a production classifier could tolerate?**

If the data is too noisy, the model will have a hard time classifying the test set.

The overall error rate doesn't seem to be affected in the first two scenarios (standard deviation of 8 and 32). It seems like the classifier is relatively robust to small amounts of noise. Note that noise added to the training set is a data augmentation strategy, hence this result is not surprising.

**Do you think that Gaussian noise is an appropriate model for real-world noise if the characters were acquired by standard digital photography? If so, in what situations? How would you compensate for it?**

The real-world noise in images acquired by standard digital photography is mainly due to the sensor of digital cameras. In this case, the images will contain random variation of brightness or color. In this situation, Gaussian noise is an appropriate model.

Various preprocessing are available to filter noise on images. Prior to use any classification method, one could use a filter to compensate for noise. We could also adapt the network accordingly.  

What we need to pay attention to is overfitting, especially we need to make sure the network is not fitting the noise. In order to do that, we could use the Dropout method or the Early Stopping method.

**Is the accuracy of certain classes affected more by image noise than others? Why?**

At first glance, it doesn't seem to be the case. You can see below the graph displaying the test error rate for the different classes. 
However, to confirm this hypothesis, it would be good to do the following experiment: Run a good number of time the training and fitting part (100 times at least) and display the boxplots (with mean and standard deviation of the error rate for each class). This experiment will need to be run twice, one time on the original dataset and one time on the modified dataset with additional noise. The two figures could then be compared. 
The modified dataset would be modified only once at the beginning

![Test Error](../output/figs/output_3/test_error.png)

## Part 3: Add Noise to the Labels

In this part, I will replace randomly a percentage of the labels in the training set. The percentage of random labels is respectively 5%, 15% and 50%.
Below you can see a graph displaying the error rate in function of the percentage of random labels in the training set.

![Test Error](../output/figs/label_noise.png)

**How important are accurate training labels to classifier accuracy?**

Accurate training labels are really important to a classifier because we optimize the output based on those labels.

As shown on the graph above, the test error doesn't drastically decrease. However, when looking only at the accuracy on the training set (shown below) one would conclude that the fit is poor, which is an issue.

|Percentage|Train Acc|Test Acc|
|-----|-----|-----|
|5%   |90.87|98.51|
|15%  |82.05|98.39|
|50% |51.37|97.62|

We see in the table that while the training accuracy drastically decreases, the test accuracy doesn't.
- it seems like the network is relatively "robust" to the noise in the labels. The features built in the network are still relevant. However the loss function isn't because we don't know what to fit to. We know that some labels are not correct.
- it's important to note that if we don't have any clean dataset, we don't have any way to measure the accuracy of the model, which is indeed a major concern.

**How would you compensate for label noise? Assume you have a large budget available but you want to use it as efficiently as possible.**

I see two ways of doing so, either by relabeling existing data or by finding new data, relabeling seems to be easier to realise than finding new data.
If you don't have enough money to relabel the entire dataset, then I would begin by relabeling only one part which would serve as a test set and would allow to have an estimation of the error rate.
(When doing model selection, it will also be really important to have a validation set to have a fair comparison of the different models tested.)

**How would you quantify the amount of label noise if you had a noisy data set?**

I would compare the accuracy of the test set with the accuracy of the training set. We can see in the table above that this difference seems to be correlated with the percentage of random labels in the training set.
In my answeer, I assumed that we have access to a test set which is clean.

**If your real-world data had both image noise and label noise, which would you be more concerned about? Which is easier to compensate for?**

I would be more concerned about label noise. It's a crucial need to be able to estimate properly the efficiency of a model. If you don't have clean labels you will have a hard time knowing if your model is fitting properly your data.
It is easier to compensate for image noise, there are many preprocessing methods and many ways to improve networks to compensate account image noise.

Although, I did some research on the topic and found that some papers focus on developping loss functions to account for a certain percentage of noise in the labels.
