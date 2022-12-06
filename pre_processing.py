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
    ''' test_dict = {'mild': [...], 'moderate': [...], 'none': [...], 'very_mild': [...] }
                      index len : 179, 12, 640, 448, (sum = 1279)
                      each array is (208, 176) => (208, 176)
    '''
    classes = ['mild', 'moderate', 'none', 'very_mild']
    # test_dict, train_dict = {'mild': [], ... }, { ... , 'very_mild': []}
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

def arrayify():
    # for i, arr in enumerate(test_dict.values()):
    #     print(i, len(arr))
    print("|test_image_mild_0|.shape =", test_dict['mild'][0].shape)
    # nparr = np.array([])
    # nparr = np.empty(shape=(0, 0), dtype=float, order='C')
    matrix = []
    for matrix in test_dict.values():
        for i, arr in enumerate(matrix):
            if i == 1: print(arr.shape)
            # nparr = np.append(nparr, arr, axis=None)
            np.stack(nparr, arr)
            # matrix.append(arr)
    print(nparr.shape)
    print(nparr[0].shape)
    np.array(matrix)
    print(matrix.shape)
    return nparr


if __name__ == '__main__':
    ## Test loading, normalizing, and plotting image data
    test_dict, train_dict = traintest_data(debug=False)
    normalize(test_dict)

    arrayify()

    ## DEBUG - PLOT BRAIN SCAN
    # for vec in list(test_dict.values())[0][0]: print(vec)
    # pltimg(list(test_dict.values())[0][0])

    ## DEBUG - CHECK LENGTH OF DATASETS
    # for value in test_dict.values():
    #     print(len(value))
    # for value in train_dict.values():
    #     print(len(value))
