import tensorflow as tf
from tensorflow import keras


def vgg():
    # Define the input layer
    input_layer = keras.Input(shape=(224, 224, 3))
    # Define the output layer
    output_layer = keras.layers.Dense(4, activation='softmax')(input_layer)
    # Define the first block of convolutional and pooling layers
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Define the second block of convolutional and pooling layers
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Define the third block of convolutional and pooling layers
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Define the fourth block of convolutional and pooling layers
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Define the fifth block of convolutional and pooling layers
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
    # Flatten the output of the convolutional layers
    x = keras.layers.Flatten()(x)
    # Define the fully-connected layers
    x = keras.layers.Dense(4096, activation='relu')(x)
    x = keras.layers.Dense(4096, activation='relu')(x)
    # Combine the output of the fully-connected layers with the output layer
    output_layer = keras.layers.Dense(4, activation='softmax')(x)

# Create the model object
model = keras.Model(inputs=input_layer, outputs=output_layer)
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

