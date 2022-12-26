import tensorflow       as tf
from tensorflow.keras   import datasets, layers, models
from pre_processing     import gen_data

# Load datasets
train_images, test_images, train_labels, test_labels = gen_data()

# MODEL - CONVOLUTIONAL LAYERS
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(208, 176, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# MODEL - DENSE LAYERS
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))
# MODEL - SUMMARY
model.summary()
# MODEL - COMPILE AND FIT
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=32, epochs=10,
                    validation_data=(test_images, test_labels),)
# MODEL - TEST
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)



