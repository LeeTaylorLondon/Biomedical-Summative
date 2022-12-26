# Author: Lee Taylor, ST Number: 190211479
import  time
import  tensorflow.keras as keras
import  sklearn.metrics  as metrics
import  numpy            as np
from    matplotlib                          import pyplot as plt
from    tensorflow.keras.applications.vgg16 import VGG16
from    tensorflow.keras.optimizers         import Adamax, SGD
from    tensorflow.keras.layers             import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from    tensorflow.keras                    import models
from    pre_processing                      import gen_data, read_nparrays


def create_model(input_shape, flu=1024, opt=Adamax(), num_classes=4):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(flu, activation='relu')(x) # Final Layer Unitss - FLU
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    # The rest of this cell is common to both defining the full architecture or using a pre-trained one
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    # model.summary()
    return model

def traintest_model(model, x_train, y_train, x_test, y_test,
                    batch_size=2, epochs=3):
    start = time.time() # Start timer
    # Train model
    print(x_train.shape, y_train.shape)
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1)
    print(f"\nTIME TAKEN: {time.time() - start}")
    print(f"TESTING ACC.: {model.evaluate(x_test, y_test)[1]}\n")


if __name__ == '__main__':
    ''' To achieve fast testing of different hyper-parameters this program 
    must be ran in the console, therefore gen_data(...) does not need to be re-ran.
     
    >>> (1279, 208, 176, 3) : test  images
    >>> (5121, 208, 176, 3) : train images
    >>> (1279, 4)           : test  labels
    >>> (5121, 4)           : train labels
    '''

    test_images, test_labels, train_images, train_labels = read_nparrays()
    x_train_, y_train_, x_test_, y_test_ = train_images, train_labels, test_images, test_labels

    ishp = (208, 176, 3) # ishp = [I]nput_[SH]a[P]e
    print(f">>> Created model(s)")
    # models_flu = [create_model(input_shape=ishp, flu=512), create_model(input_shape=ishp, flu=1024),
    #               create_model(input_shape=ishp, flu=2048), create_model(input_shape=ishp, flu=4096),]
    flu = [512, 1024, 2048, 4096]

    for flu_ in flu:
        print(f">>> Train-testing model")
        model = create_model(input_shape=ishp, flu=flu_)
        traintest_model(model, x_train_, y_train_, x_test_, y_test_)
        del model

    # m_vgg = create_model((208, 176, 3))

    # train_images, test_images, train_labels, test_labels = gen_data(dim=3, debug=True)
    # data = [train_images, test_images, train_labels, test_labels]
    # for d in data: print(d.shape)
    # # train_images, train_labels, test_images, test_labels = gen_data_fake(dim=3, debug=True)
    # x_train_, y_train_, x_test_, y_test_ = train_images, train_labels, test_images, test_labels
    # numpy.save("train_images", train_images)
    # numpy.save("test_images", test_images)
    # numpy.save("train_labels", train_labels)
    # numpy.save("test_labels", test_labels)
    # print(f">>> Created train & test data")

    # y_train_ = np.asarray(train_labels).astype('float32').reshape((-1, 1))
    # y_test_ = np.asarray(test_labels).astype('float32').reshape((-1, 1))
    # print(f">>> Reshaped label data")

    # print(f"{x_train_.shape}, {x_test_.shape}")
    # print(f"{x_train_.shape}, {x_test_.shape}")
    # print(f">>> Dataset shapes ^")

    # print(f">>> Train-testing model")
    # traintest_model(m_vgg, x_train_, y_train_, x_test_, y_test_)
