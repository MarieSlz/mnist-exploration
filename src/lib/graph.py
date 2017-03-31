'''
Figures and graphs ploting functions
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

def plot_sample(x_train):
    '''
    Plot a subsamples of 4 MNIST images
    '''
    plt.subplot(221)
    plt.imshow(np.reshape(x_train[0], (-1,28)), cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    plt.imshow(np.reshape(x_train[1], (-1,28)), cmap=plt.get_cmap('gray'))
    plt.subplot(223)
    plt.imshow(np.reshape(x_train[2], (-1,28)), cmap=plt.get_cmap('gray'))
    plt.subplot(224)
    plt.imshow(np.reshape(x_train[3], (-1,28)), cmap=plt.get_cmap('gray'))
    plt.savefig('../output/figs/sample_mnist_images.png')
    plt.close()

def save_cnf_matrix(cnf_matrix, folder):
    '''
    Draw and Save the confusion Matrix
    '''
    cnf = pd.DataFrame(cnf_matrix, index = [i for i in "0123456789"],
                         columns = [i for i in "0123456789"])
    fig = plt.figure(figsize = (10,7))
    plot = sn.heatmap(cnf, annot=True, fmt='d')
    plot.set(xlabel='Predicted Label', ylabel='True Label')
    path = os.path.join("../output/figs",folder)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join("../output/figs", folder, "confusion_matrix.png")
    fig.savefig(path)
    plt.close()

def error(score, folder):
    '''
    Draw and Save the Test Error for each class
    '''
    #Test Error
    error_df = pd.DataFrame({'class': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                             'error': list((1-score[0])*100)})
    sn.set_style("darkgrid")
    fig = plt.figure(figsize=(10, 7))
    error_plot = sn.barplot(x=error_df["class"],y=error_df["error"],
                            palette="muted")
    error_plot.set_xlabel("Class")
    error_plot.set_ylabel("Error Rate (%)")
    error_plot.set_title("Error Rate per Class")
    error = error_plot.get_figure()
    path = os.path.join("../output/figs",folder)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join("../output/figs", folder, "test_error.png")
    error.savefig(path)
    plt.close()

def precision(score, folder):
    '''
    Draw and Save the Precision for each class
    '''
    # Precision
    precision_df = pd.DataFrame({'class': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                 'precision': list(score[0]*100)})
    sn.set_style("darkgrid")
    fig = plt.figure(figsize=(10, 7))
    precision_plot = sn.barplot(x=precision_df["class"],y=precision_df["precision"],
                            palette="muted")
    precision_plot.set_xlabel("Class")
    precision_plot.set_ylabel("Precision (%)")
    precision_plot.set_title("Precision per Class")
    precision = precision_plot.get_figure()
    path = os.path.join("../output/figs",folder)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join("../output/figs", folder, "precision.png")
    precision.savefig(path)
    plt.close()

def recall(score, folder):
    '''
    Draw and Save the Recall for each class
    '''
    # Recall
    recall_df = pd.DataFrame({'class': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                 'recall': list(score[1]*100)})
    sn.set_style("darkgrid")
    fig = plt.figure(figsize=(10, 7))
    recall_plot = sn.barplot(x=recall_df["class"],y=recall_df["recall"],
                            palette="muted")
    recall_plot.set_xlabel("Class")
    recall_plot.set_ylabel("Precision (%)")
    recall_plot.set_title("Recall per Class")
    recall = recall_plot.get_figure()
    path = os.path.join("../output/figs",folder)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join("../output/figs", folder, "recall.png")
    recall.savefig(path)
    plt.close()

def loss(losses, num_batchs, folder):
    '''
    Draw and Save the Loss for the first batch of training
    '''
    losses = losses[:num_batchs]
    batch_number = [i+1 for i in range(num_batchs)]
    fig = plt.figure(figsize=(10, 7))
    plt.plot(batch_number, losses, linewidth=5)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Batches')
    path = os.path.join("../output/figs",folder)
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join("../output/figs", folder, "loss_first_epoch.png")
    plt.savefig(path)
    plt.close()

def output_graphs(y_test, y_pred, history, NUM_BATCHS, folder):
    '''
    Draw and Save a banch of differnt grapsh summarising the results
    Confusion matrix, Loss during the first batch, Precision, Recall and Test Error per class
    '''
    print("\n[INFO] Saving and Ploting results")
    # Confusion Matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    save_cnf_matrix(cnf_matrix, folder)
    # Prediction Per Class - Multiclass classification
    score = precision_recall_fscore_support(y_test, y_pred,
                                            average=None,
                                            labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    error(score, folder)
    precision(score, folder)
    recall(score, folder)
    # Loss per batch
    loss(history.losses, NUM_BATCHS, folder)

def std_error(std, error):
    '''
    Draw and Save the Global Test Error in function of the standard deviation of the noise added to the images
    '''
    plt.figure()
    plt.scatter(std, error)
    plt.xlabel("Standard Deviation")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate for Different Levels of Noise")
    plt.savefig("../output/figs/image_noise.png")
    plt.close()

def per_error(per, error):
    '''
    Draw and Save the Global Test Error in function of the percentage of random labels
    '''
    plt.figure()
    plt.scatter(per, error)
    plt.xlabel("Percentage of Noisy Labels (%)")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate for Different Percentage of Randomn Labels")
    plt.savefig("../output/figs/label_noise.png")
    plt.close()
