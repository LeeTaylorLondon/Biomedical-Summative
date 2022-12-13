# Author: Lee Taylor, ST Number: 190211479
import  time
import  tensorflow.keras as keras
import  sklearn.metrics  as metrics
import  numpy            as np
from    matplotlib                          import pyplot as plt
from    tensorflow.keras.applications.vgg16 import VGG16
from    tensorflow.keras                    import models
from    tensorflow.keras.optimizers         import Adamax, SGD
from    pre_processing                      import gen_data
from    tensorflow.keras.layers             import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D


def create_model(input_shape, opt=Adamax(), num_classes=4):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(4096, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    # The rest of this cell is common to both defining the full architecture or using a pre-trained one
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    # model.summary()
    return model

def traintest_model(model, batch_size=32, epochs=3):
    start = time.time() # Start timer
    # Train model
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    print(f"\nTIME TAKEN: {time.time() - start}")
    print(f"TESTING ACC.: {model.evaluate(x_test, y_test)[1]}\n")


if __name__ == '__main__':
    m_vgg = create_model((208, 176, 3))

    train_images, test_images, train_labels, test_labels = gen_data(dim=3, debug=True)
    x_train, y_train, x_test, y_test = train_images, train_labels, test_images, test_labels

    y_test = np.asarray(test_labels).astype('float32').reshape((-1, 1))

    print(f"{x_train.shape}, {x_test.shape}")
    print(f"{x_train.shape}, {x_test.shape}")

    traintest_model(m_vgg)
