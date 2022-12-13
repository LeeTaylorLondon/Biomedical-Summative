# Author: Lee Taylor, ST Number: 190211479
import  cv2
import  matplotlib.pyplot   as plt
import  matplotlib.image    as mpimg
import  numpy               as np
from    os                  import walk
from    typing              import List


# img = mpimg.imread('Data/images/test/mild/26.jpg')
# print(f"img shape: {img.shape}")
# >>> (208, 176)
# filenames = next(walk('Data/images/test/mild'), (None, None, []))[2]  # [] if no file
# print(filenames)
# >>> ['26.jpg', '26_19.jpg', '26_20.jpg', '26_21.jpg', '26_22.jpg', ..., '32_9.jpg']

def pltimg(i):
    plt.imshow(i)
    plt.show()

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
            filenames = next(walk(f'../Data/{datatype}/{class_}'), (None, None, []))[2]
            if debug: print(f"DEBUG filenames:{filenames}")
            for fn in filenames:
                arr.append(mpimg.imread(f'../Data/{datatype}/{class_}/{fn}'))
    # Debug
    if debug:
        for i,arr in enumerate(test_dict.values()): print(f"traintest_data() -> len(dict_.arr[{i}]) = {len(arr)}")
    return train_dict, test_dict

def normalize(dict, debug=False):
    dict = dict.copy()
    for arr in dict.values():
        for i,v in enumerate(arr):
            arr[i] = v/255
    return dict

def dict_to_nparray(dict_, debug=False):
    npm = np.empty((0, 208, 176))
    for vec in dict_.values():
        vec = np.array(vec)
        npm = np.append(npm, vec, axis=0)
    if debug: print(npm.shape)
    return npm

def dict_to_nparray_dim(dict_, dim=3, debug=False):
    npm = np.empty((0, 208, 176, dim))
    for vec in dict_.values(): # >>> [np.array(208, 176), ...]
        for i,img in enumerate(vec):
            img_ = np.array([cv2.merge((img, img, img))])
            # print(f"npm.shape={npm.shape} img_.shape={img_.shape}")
            npm = np.append(npm, img_, axis=0)
    if debug: print(npm.shape)
    return npm

def images_to_labels(dict_, debug=False):
    labels = np.empty(0)
    for i,arr in enumerate(dict_.values()):
        temp_labels = np.ones(shape=(len(arr))) * (i)
        labels = np.append(labels, temp_labels)
    if debug: print(labels, labels.shape)
    return labels

def gen_data(dim, debug=False):
    test_dict, train_dict = traintest_data(debug=False)
    if len(list(test_dict.values())[0]) == 0: raise ValueError
    train_images   = normalize(train_dict)
    test_images    = normalize(test_dict)
    if dim == None:
        train_images   = dict_to_nparray(train_images, debug=debug)
        test_images    = dict_to_nparray(test_images, debug=debug)
    else:
        train_images   = dict_to_nparray_dim(train_images, debug=debug)
        test_images    = dict_to_nparray_dim(test_images, debug=debug)
    train_labels   = images_to_labels(train_dict)
    test_labels    = images_to_labels(test_dict)
    return train_images, test_images, train_labels, test_labels


if __name__ == '__main__':
    # # Generate train & test data
    # test_dict, train_dict = traintest_data()
    # normalize(test_dict)
    # train_images, test_images, labels = arrayify()

    ## Test loading, normalizing, and plotting image data
    test_dict_, train_dict_ = traintest_data(debug=False)
    normalize(test_dict_)
    normalize(train_dict_)

    # Test functions
    test_images  = dict_to_nparray(test_dict_, debug=True)
    test_labels  = images_to_labels(test_dict_, debug=True)
    train_images = dict_to_nparray(train_dict_, debug=True)
    train_labels = images_to_labels(train_dict_, debug=True)



