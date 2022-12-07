# Author: Lee Taylor, ST Number: 190211479
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from os import walk


def pltimg(i):
    plt.imshow(i)
    plt.show()

# img = mpimg.imread('Data/images/test/mild/26.jpg')
# print(f"img shape: {img.shape}")
# >>> (208, 176)
# filenames = next(walk('Data/images/test/mild'), (None, None, []))[2]  # [] if no file
# print(filenames)
# >>> ['26.jpg', '26_19.jpg', '26_20.jpg', '26_21.jpg', '26_22.jpg', ..., '32_9.jpg']

def traintest_data(debug=False):
    """ test_dict = {'mild': [...], 'moderate': [...], 'none': [...], 'very_mild': [...] }
                      index len : 179, 12, 640, 448, (sum = 1279)
                      each array is (208, 176) => (208, 176) """
    classes = ['mild', 'moderate', 'none', 'very_mild']
    test_dict, train_dict = ({}), ({})
    for class_ in classes:
        test_dict.update({class_:[]})
        train_dict.update({class_:[]})
    # Populate each array with image values
    for datatype, dict_ in zip(['test', 'train'], [test_dict, train_dict]):
        for class_, arr in zip(classes, dict_.values()):
            filenames = next(walk(f'Data/images/{datatype}/{class_}'), (None, None, []))[2]
            for fn in filenames:
                arr.append(mpimg.imread(f'Data/images/{datatype}/{class_}/{fn}'))
    # Debug
    if debug:
        for arr in test_dict.values(): print(len(arr))
    return test_dict, train_dict

def normalize(dict, debug=False):
    for arr in dict.values():
        for i,v in enumerate(arr):
            arr[i] = v/255

def dict_to_nparray(dict_, debug=False):
    npm = np.empty((0, 208, 176))
    for vec in dict_.values():
        vec = np.array(vec)
        npm = np.append(npm, vec, axis=0)
    if debug: print(npm.shape)
    return npm

def images_to_labels(dict_, debug=False):
    labels = np.empty(0)
    for i,arr in enumerate(dict_.values()):
        temp_labels = np.ones(shape=(len(arr))) * (i + 1)
        labels = np.append(labels, temp_labels)
    if debug: print(labels, labels.shape)
    return labels


if __name__ == '__main__':
    ## Test loading, normalizing, and plotting image data
    test_dict_, train_dict_ = traintest_data(debug=False)
    normalize(test_dict_)
    normalize(train_dict_)

    test_images  = dict_to_nparray(test_dict_, debug=True)
    test_labels  = images_to_labels(test_dict_, debug=True)
    train_images = dict_to_nparray(train_dict_, debug=True)
    train_labels = images_to_labels(train_dict_, debug=True)


