# Author: Lee Taylor, ST Number: 190211479
import  cv2
import  time
import  pickle
import  gzip
import  matplotlib.pyplot   as plt
import  matplotlib.image    as mpimg
import  numpy               as np
from    os                  import walk
from    typing              import List
from    sklearn.utils       import shuffle


# img = mpimg.imread('Data/images/test/mild/26.jpg')
# print(f"img shape: {img.shape}")
# >>> (208, 176)
# filenames = next(walk('Data/images/test/mild'), (None, None, []))[2]  # [] if no file
# print(filenames)
# >>> ['26.jpg', '26_19.jpg', '26_20.jpg', '26_21.jpg', '26_22.jpg', ..., '32_9.jpg']

np.random.seed(42)

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
    for j,vec in enumerate(dict_.values()): # >>> [np.array(208, 176), ...]
        print(f"dict_.values()[{j+1}] out of [4]")
        for i,img in enumerate(vec):
            print(f"d_t_npa_dim(...)-> img:{i}")
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

def images_to_labels_dim(dict_, debug=False):
    """  """
    labels = np.empty((0, 4))
    for i,arr in enumerate(dict_.values()):
    # for i in range(50):
        temp_labels = np.array([np.zeros(4, dtype=float) for _ in range(len(arr))])
        print(f"temp_labels.shape = {temp_labels.shape}")
        for inner_arr in temp_labels: inner_arr[i] = 1.0
        labels = np.append(labels, temp_labels, axis=0)
    if debug: print(labels, labels.shape)
    return labels

def gen_data(dim=None, debug=False):
    timer = time.time()
    test_dict, train_dict = traintest_data(debug=False)
    print(f"traintest_data(): {time.time() - timer}s")

    timer = time.time()
    if len(list(test_dict.values())[0]) == 0: raise ValueError
    train_images   = normalize(train_dict)
    test_images    = normalize(test_dict)
    print(f"normalization(): {time.time() - timer}s")
    if dim == None:
        train_images   = dict_to_nparray(train_images, debug=debug)
        test_images    = dict_to_nparray(test_images, debug=debug)
    else:

        timer = time.time()
        train_images   = dict_to_nparray_dim(train_images, debug=debug)
        print(f"d_to_nparr(train): {time.time() - timer}s")

        timer = time.time()
        test_images    = dict_to_nparray_dim(test_images, debug=debug)
        print(f"d_to_nparr(test): {time.time() - timer}s")

    timer = time.time()
    train_labels = images_to_labels_dim(train_dict)
    print(f"i_to_l(train): {time.time() - timer}s")

    timer = time.time()
    test_labels = images_to_labels_dim(test_dict)
    print(f"i_to_l(train): {time.time() - timer}s")
    return train_images, test_images, train_labels, test_labels

def save_nparrays(datasets_):
    for d in dataset_:
        np.save('array.np', )
    pass

def read_nparrays():
    dir_, rv = '../Data/pre_pro/', []
    files    = ['train_images', 'train_labels', 'test_images', 'test_labels']
    for file in files: rv.append(np.load(f"{dir_+file}.npy"))
    return rv
    # nparrays = [np.load(dir_ + fn) for fn in files]
    # return [np.load(dir_ + fn) for fn in files]

def to_pk1gz(obj=None, fd=None):
    # Create an object to serialize
    if obj == None: obj = {'a': 1, 'b': 2, 'c': 3}
    if fd  == None: fd  = 'obj.pk1.gz'
    # Open a .pk1.gz file in write mode
    with gzip.open(fd, 'wb') as f:
        # Serialize the object and write it to the file
        pickle.dump(obj, f)
    print(f"function to_pk1gz(...) finished!")

def shuffle_(dict_):
    # Shuffle the training set
    x_train, y_train = shuffle(dict_['train_images'], dict_['train_labels'], random_state=42)
    # Shuffle the test set
    x_test, y_test = shuffle(dict_['test_images'], dict_['test_labels'], random_state=42)
    del dict_
    return x_train, y_train, x_test, y_test

def from_pk1gz(fd=None, debug=False):
    if fd == None: fd = 'obj.pk1.gz'
    # Open the .pk1.gz file in read mode
    with gzip.open(fd, 'rb') as f:
      # Load the data from the file and deserialize it
      obj = pickle.load(f)
    if debug: print(f"from_pk1gz(...) -> {obj}")
    # Returns tuple = (x_train, y_train, x_test, y_test)
    return shuffle_(obj)

if __name__ == '__main__':
    """ Create .npz files of the arrays to be read
     *Create 1 channel arrays
     *Create 3 channel arrays
    """
    # # Create train and test dictionaries
    # train_dict, test_dict = traintest_data(debug=False)
    # train_dict, test_dict = normalize(train_dict), normalize(test_dict)
    # # Create train and test labels (1 dimensional)
    # train_labels = images_to_labels(train_dict)
    # test_labels  = images_to_labels(test_dict)
    # # Convert X dictionaries to nparrays
    # train_images = dict_to_nparray(train_dict)
    # test_images  = dict_to_nparray(test_dict)
    #
    # # Create container for datasets
    # # X_train, y_train, X_test, y_test
    # data = {"train_images": train_images, "train_labels": train_labels,
    #         "test_images": test_images, "test_labels": test_labels}
    # # Test pk1gz functionality
    # to_pk1gz(data, "../Data/1d.pk1.gz")
    # from_pk1gz("../Data/1d.pk1.gz")
    #
    # Load 3 channel data from numpy array files
    test_images_, test_labels_, train_images_, train_labels_ = read_nparrays()
    # Create container for datasets (3 channels)
    data3 = {"train_images": train_images_, "train_labels": train_labels_,
             "test_images": test_images_, "test_labels": test_labels_}
    # Test pk1gz functionality
    to_pk1gz(data3, "../Data/3d.pk1.gz")
    from_pk1gz("../Data/3d.pk1.gz")

    """ General testing """
    # arrays = read_nparray()
    # for arr in arrays: print(arr)

    # # Generate train & test data
    # test_dict, train_dict = traintest_data()
    # normalize(test_dict)
    # train_images, test_images, labels = arrayify()

    ## Test loading, normalizing, and plotting image data
    # test_dict_, train_dict_ = traintest_data(debug=False)
    # print(test_dict_, train_dict_)
    # normalize(test_dict_)
    # normalize(train_dict_)
    #
    # # Test functions
    # test_images  = dict_to_nparray(test_dict_, debug=True)
    # test_labels  = images_to_labels(test_dict_, debug=True)
    # train_images = dict_to_nparray(train_dict_, debug=True)
    # train_labels = images_to_labels(train_dict_, debug=True)

    # def gen_data_fake(dim, debug=True):
    #     # train_images, test_images, train_labels, test_labels
    #     # train_images.shape = (1279, 208, 176, 3)
    #     test_dict_, train_dict_ = traintest_data(debug=False)
    #     x_train = np.ones(shape=(50, 208, 176, 3))
    #     # y_train = np.ones(shape=(50, 4))
    #     y_train = images_to_labels_dim(train_dict_)
    #     x_test = np.ones(shape=(25, 208, 176, 3))
    #     # y_test  = np.ones(shape=(25, 4))
    #     y_test = images_to_labels_dim(test_dict_)
    #     datasets = [x_train, y_train, x_test, y_test]
    #     if debug:
    #         for dataset in datasets: print(dataset.shape)
    #     return x_train, y_train, x_test, y_test



