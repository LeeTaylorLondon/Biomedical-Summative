import  tensorflow                          as      tf
from    tensorflow                          import  keras
from    pre_processing                      import  from_pk1gz
from    tensorflow.keras.layers             import  Dense, Flatten, Dropout
from    tensorflow.keras.applications.vgg16 import  VGG16
from    itertools                           import product
import  matplotlib.pyplot                   as plt
import  numpy                               as np
import  random


# Set random seed for reproducibility, load datasets, define input shape
tf.random.set_seed(42)
# data        = from_pk1gz("../Data/3d.pk1.gz")
x_train, y_train, x_test, y_test = from_pk1gz("../Data/3d.pk1.gz")
input_shape = (208, 176)

def train_test_model(model):
    # Train the model
    model.fit(x=data['train_images'], y=data['train_labels'], epochs=1, batch_size=512)
    # Evaluate the model on the test set
    results = model.evaluate(x=data['test_images'], y=['test_labels'])
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    return results

def create_lenet():
    model = keras.Sequential()
    # Conv2D layer 6 filters, kernel size (5, 5), ReLU activation
    model.add(keras.layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
    # AveragePooling2D layer pool size (2, 2)
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    # Conv2D layer 16 filters, kernel size (5, 5)
    model.add(keras.layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    # AveragePooling2D layer pool size (2, 2)
    model.add(keras.layers.AveragePooling2D(pool_size=(2, 2)))
    # Flatten the output
    model.add(keras.layers.Flatten())
    # Fully-connected layer 120 units, ReLU activation
    model.add(keras.layers.Dense(120, activation='relu'))
    # Fully-connected layer 84 units, ReLU activation
    model.add(keras.layers.Dense(84, activation='relu'))
    # Fully-connected output layer, 10 units and softmax activation
    model.add(keras.layers.Dense(10, activation='softmax'))
    # Compile the model
    model.compile(loss=tf.losses.CategoricalCrossentropy(), optimizer=tf.optimizers.Adam(), metrics=['accuracy'])
    return model

def create_lenet_variation(learning_rate=0.001, filter_size=(5, 5),
                           subsampling_layer=keras.layers.AveragePooling2D,
                           layers=2, filters=[6, 16], activation='relu',
                           d_layers=2, units=[120, 84]):
    """ This function creates variations of the base model """
    # Define base model to add to
    model    = keras.Sequential()
    # Stores last filter value for Conv2D layer
    filter_x = None
    for i in range(layers):
        # try-except accounts for filter values from list
        # ... or dynamically calculate them
        try:
          filter_ = filters[i]
          filter_x = filter_
        except IndexError as e:
          filter_ = filter_x + 10
          filter_x = filter_
        # Add C layer
        model.add(keras.layers.Conv2D(filter_, kernel_size=filter_size,
                                      activation='relu',
                                      input_shape=input_shape))
        # Add S layer
        model.add(subsampling_layer(pool_size=(2, 2)))
    # Flatten the output & add fully connected layers
    model.add(keras.layers.Flatten())
    for d_ in range(d_layers):
        model.add(keras.layers.Dense(units[d_], activation=activation))
    # Output layer, 10 units and softmax activation
    model.add(keras.layers.Dense(4, activation='softmax'))
    # Compile the model
    model.compile(loss=tf.losses.CategoricalCrossentropy(),
                  optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy'])
    return model

def gridsearch(model, hyperparams, X_train, y_train, X_test, y_test, verbose=True,
               store_models=False):
    """ Perform grid search on a given model with a given set of hyperparameters.
    Using a dictionary containing the hyperparameters to be searched over. The keys
    should be the names of the hyperparameters and the values should be lists
    of values to try.
    """
    # Error check
    for arr in hyperparams.values():
      if len(arr) == 0:
        raise TypeError("A hyperparameter is empty == [], most likely 'filter'.")
    # Initialize lists to store the results of each model
    # Scores stores testing accuracies only whereas score_ stores four metrics
    params, scores, score_, m_inst = [], [], [], []
    # Get the names of the hyperparameters
    param_names = list(hyperparams.keys())
    # Create a list of lists of values to try for each hyperparameter
    param_values = list(hyperparams.values())
    # Use itertools.product to generate all combinations of hyperparameter values
    for values in product(*param_values):
        # Create a dictionary mapping hyperparameter names to values
        param_dict = dict(zip(param_names, values))
        # Ensure filters corresponds to layers
        if param_dict['layers'] > len(param_dict['filters']):
            # i.e 3 > 2
            pdf = param_dict['filters']
            while param_dict['layers'] != len(param_dict['filters']):
                param_dict['filters'].append(param_dict['filters'][-1] +
                                             abs(pdf[-1] - pdf[-2]))
                # [6, 16] -> [6, 16, 26]
        # Extract batch_size and epochs from the dictionary
        batch_size = param_dict.pop("batch_size")
        epochs = param_dict.pop("epochs")
        # Create the model with the current set of parameters
        model_instance = model(**param_dict)
        # Add epochs and batch size back for recording
        param_dict.update({"batch_size": batch_size})
        param_dict.update({"epochs": epochs})
        # Fit & test model, record train acc., & loss, & test acc.
        if verbose:
          for i,v in param_dict.items():
            print(f"{i:18} = {v}")
        print("Training model...")
        history       = model_instance.fit(X_train, y_train,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           verbose=False)
        training_acc  = round(history.history['accuracy'][-1], 6)
        training_loss = round(history.history['loss'][-1], 6)
        test_loss, test_acc = model_instance.evaluate(X_test, y_test, verbose=0)
        if verbose:
          print(f"Training Accuracy: {training_acc}\n"
                f"Testing Accuracy:  {test_acc}\n"
                f"Training Loss: {training_loss}\n"
                f"Testing Loss:  {test_loss}\n")
        # Store the parameters for the model
        score_.append([test_acc, test_loss, training_acc, training_loss])
        scores.append(test_acc)
        params.append(param_dict)
        # Update best model stored object instance
        if test_acc > max(scores):
          m_inst = model_instance
        if store_models: m_inst.append(model_instance)
        else: del model_instance
    # Calculate best index
    best_index = np.argmax(scores)
    if store_models:
      return params[best_index], scores[best_index], \
              best_index, m_inst[best_index]
    # Return the best parameters and the best score
    return params[best_index], scores[best_index], best_index, m_inst

""" Actual usagev - optimize; layers """
# Define hyperparams to search over
hyperparams = {"learning_rate"    : [0.001, 0.002, 0.003],
               "filter_size"      : [(x, x) for x in range(3, 6)],
               "subsampling_layer": [keras.layers.AveragePooling2D,
                                     keras.layers.MaxPooling2D],
               "epochs"           : [1, 5],
               "batch_size"       : [128],
               "layers"           : [2],
               "filters"          : [
                   [6, 16],
                   [30, 50]
                                     ],
               "activation"       : ['relu'],
               "d_layers"         : [2],
               "units"            : [[120, 84]]}
               # "d_layers"         : [4],
               # "units"            : [[120, 84, 64, 32],
               #                       [120, 120, 120, 120],
               #                       [32, 64, 84, 120]
               #                       ]}

# Perform grid search
gridsearch(create_lenet_variation, hyperparams,
           x_train, y_train, x_test, y_test, verbose=True)