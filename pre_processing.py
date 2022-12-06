# Author: Lee Taylor, ST Number: 190211479
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    classes = ['mild', 'moderate', 'none', 'very_mild']
    # test_dict, train_dict = {'mild': [], ... }, { ... , 'very_mild': []}
    test_dict, train_dict = ({}), ({})
    for class_ in classes:
        test_dict.update({class_:[]})
        train_dict.update({class_:[]})
    # Populate each array with image values
    for class_, test_arr in zip(classes, test_dict.values()):
        filenames = next(walk(f'Data/images/test/{class_}'), (None, None, []))[2]
        for fn in filenames:
            test_arr.append(mpimg.imread(f'Data/images/test/{class_}/{fn}'))
    # Debug
    if debug:
        for arr in test_dict.values(): print(len(arr))
    return test_dict, train_dict

def normalize(dict, debug=False):
    for arr in dict.values():
        for i,v in enumerate(arr):
            arr[i] = v/255

def traintest_labels():
    pass


if __name__ == '__main__':
    # Test loading, normalizing, and plotting image data
    test_dict, train_dict = traintest_data()
    normalize(test_dict)
    for vec in list(test_dict.values())[0][0]: print(vec)
    pltimg(list(test_dict.values())[0][0])
    pass