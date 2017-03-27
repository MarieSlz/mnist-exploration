# Exploration of the MNIST dataset

## Project Summary:
In this project, my goal is to train a convolutional neural network classifier on the MNIST data set and explore how image noise and label noise affect classifier accuracy.

## Project Organisation:
My answers are available [here](doc/answers.md)

**Part 1: Classifier on original data**
In this first part, I will choose and train a convolutional neural network on the MNIST dataset. Among others, the questions I will answer are the followings:
- What is your test set error rate?
- What is the test set error rate for each class? Are some classes more challenging than others to distinguish from each other? Why?
- Based only on information gathered in the first epoch of training, do you think that the model would benefit from more training time? Why?
- Besides training for a longer time, what would you do to improve accuracy?

**Part 2: Image Noise Impact**
In this part, I will add random Gaussian noise to the training set images. Among others, the questions I will answer are the followings:
- What is your error rate?
- What are the implications of the dependence of accuracy on noise if you were to deploy a production classifier? How much noise do you think a production classifier could tolerate?
- Do you think that Gaussian noise is an appropriate model for real-world noise if the characters were acquired by standard digital photography? If so, in what situations? How would you compensate for it?
- Is the accuracy of certain classes affected more by image noise than others? Why?

**Part 3: Label Noise Impact**
In this part, I will study the impact of label noise, I will first take 5% of the training images and randomize their labels. Among others, the questions I will answer are the followings:
- How important are accurate training labels to classifier accuracy?
- How would you compensate for label noise? Assume you have a large budget available but you want to use it as efficiently as possible.
- How would you quantify the amount of label noise if you had a noisy data set?
- If your real-world data had both image noise and label noise, which would you be more concerned about? Which is easier to compensate for?


This repo has been organized as follows.
```
proj/
├── data/
├── doc/
├── lib/
└── output/
```
