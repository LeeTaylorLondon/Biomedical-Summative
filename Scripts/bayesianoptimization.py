from skopt import BayesSearchCV
from sklearn.utils import shuffle
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from pre_processing import from_pk1gz
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import numpy as np

# Set the seed to ensure reproducibility
np.random.seed(42)

# Set random seed for reproducibility, load datasets, define input shape
tf.random.set_seed(42)
x_train, y_train, x_test, y_test = from_pk1gz("../Data/3d.pk1.gz")
input_shape = (208, 176, 3)

# Define the CNN model
def cnn_model(num_filters=16, kernel_size=3, pool_size=2, dense_size=16):
    model = Sequential()
    model.add(Conv2D(num_filters, kernel_size, activation='relu',
                    input_shape=(208, 176, 3)))
    model.add(MaxPooling2D(pool_size))
    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    # model.score = model.evaluate(x_test, y_test)[1]
    # model.get_params = params
    return model

# Set up Bayesian Optimization
opt = BayesSearchCV(estimator=KerasClassifier(build_fn=cnn_model),
                    search_spaces={'num_filters': (16, 128),
                     'kernel_size': (3, 5),
                     'pool_size': (2, 4),
                     'dense_size': (16, 128)},
                    verbose=1)

# executes bayesian optimization
_ = opt.fit(x_train, y_train)

# model can be saved, used for predictions or scoring
print(opt.score(x_test, y_test))
