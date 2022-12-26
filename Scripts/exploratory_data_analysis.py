from pre_processing import from_pk1gz
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import cv2


# pk1gz load 1D data
data = from_pk1gz("../Data/1d.pk1.gz")
X_train, y_train, X_test, y_test = data.values()

# # pk1gz load 3D data
# data = from_pk1gz("../Data/3d.pk1.gz")
# X_train3d, y_train3d, X_test3d, y_test3d = data.values()
# del data


def plot_imgs():
    labels = [x for x in range(4)]
    all_images, all_labels = [], []
    for label in labels:
        # Get the indices of all images with the current label
        indices = [i for i, y in enumerate(y_train) if y == label]
        # Select 10 random images from the list of indices
        selected_indices = random.sample(indices, 10)
        # Get the images and labels for the selected indices
        selected_images = X_train[selected_indices]
        selected_labels = y_train[selected_indices]
        # Append the selected images and labels to the all_images and all_labels lists
        all_images.extend(selected_images)
        all_labels.extend(selected_labels)
    # Make a figure with 10 rows and 10 columns
    fig, ax = plt.subplots(10, 4, figsize=(10, 4))
    # Flatten the ax array
    ax = ax.flatten()
    # Iterate over the selected images and labels and plot them
    for i, (image, label) in enumerate(zip(all_images, all_labels)):
        ax[i].imshow(image)
        # ax[i].set_title(label)
        ax[i].axis('off')
    # Show the plot
    plt.show()


def histogram():
    # Load the MRI images
    images = X_train # load the images here
    # Flatten the images into a single array of pixel values
    pixels = [image.flatten() for image in images]
    pixels = np.concatenate(pixels)
    # Plot the histogram of pixel values
    plt.hist(pixels, bins=256, range=(0.001, 1))
    plt.show()


def boxplot():
    # Flatten the images so that we have a single list of pixel values for all images
    all_pixel_values = np.concatenate([image.flatten() for image in X_train])
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Use the boxplot function to plot the pixel values
    ax.boxplot(all_pixel_values)
    # Add a title and labels
    ax.set_title("Pixel Value Distributions")
    ax.set_xlabel("Image")
    ax.set_ylabel("Pixel Value")
    # Show the plot
    plt.show()


def cls_distribution():
    # Create a list of unique labels
    labels      = np.unique(y_train)
    class_names = ['mild', 'moderate', 'none', 'very_mild']
    # Get the counts of each label in the train set
    train_counts = [np.sum(y_train == label) for label in labels]
    # Get the counts of each label in the test set
    test_counts = [np.sum(y_test == label) for label in labels]
    # Create a bar plot showing the distribution of labels in the train and test sets
    plt.bar(labels, train_counts, width=0.4, label='Train')
    plt.bar(labels + 0.4, test_counts, width=0.4, label='Test')
    plt.xticks(labels, class_names)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def averages():
    # Reshape the images into a 2D array
    images_2d = X_train.reshape(X_train.shape[0], -1)
    # Create subplot for two plots and 1 row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # Calculate and plot the full averages
    pixel_averages = np.mean(images_2d, axis=0)
    ax1.hist(pixel_averages, bins=256)
    # Calculate and plot the averages without zeros
    non_zero_means = pixel_averages[pixel_averages != 0.0]
    ax2.hist(non_zero_means, bins=100)
    # Labels and show plot
    plt.xlabel('Pixel intensity')
    plt.ylabel('Count')
    plt.show()


def std_():
    # Flatten the images into a single array of pixel values
    all_pixel_values = np.concatenate([image.flatten() for image in X_train])
    # Calculate the standard deviation of the pixel values
    std_dev = np.std(all_pixel_values)
    # Calculate the standard deviation of the pixel values in the training set
    std_dev_train = np.std(X_train.reshape(X_train.shape[0], -1), axis=1)
    # Calculate the standard deviation of the pixel values in the test set
    std_dev_test = np.std(X_test.reshape(X_test.shape[0], -1), axis=1)
    # Create a histogram of the standard deviations in the training set
    plt.hist(std_dev_train, bins=50, alpha=0.5, label='Train')
    # Create a histogram of the standard deviations in the test set
    plt.hist(std_dev_test, bins=50, alpha=0.5, label='Test')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # plot_imgs()
    # histogram()
    # boxplot()
    # cls_distribution()
    # averages()
    # std_()
    pass
