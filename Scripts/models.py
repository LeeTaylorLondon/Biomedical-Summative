import  tensorflow                          as      tf
from    tensorflow                          import  keras
from    pre_processing                      import  from_pk1gz
from    tensorflow.keras.layers             import  Dense, Flatten, Dropout
from    tensorflow.keras.applications.vgg16 import  VGG16


# Constants
data = from_pk1gz("../Data/3d.pk1.gz")
input_shape = (208, 176, 3)

def train_test_model(model):
    # Train the model
    model.fit(x=data['train_images'], y=data['train_labels'])
    # Evaluate the model on the test set
    results = model.evaluate(x=data['test_images'], y=['test_labels'])
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    return results

def vgg_(input_shape=input_shape, units=1024, activation='relu', dropout=0.5):
    # Load the VGG model
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the layers of the VGG model
    for layer in vgg_model.layers:
        layer.trainable = False
    # Add a classification head to the VGG model
    model = tf.keras.Sequential()
    model.add(vgg_model)
    model.add(Flatten())
    model.add(Dense(units, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(4, activation='softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def resnet_(input_shape=input_shape, units=1024, activation='relu', dropout=0.5):
    resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    resnet_model.trainable = False
    model = tf.keras.Sequential()
    model.add(resnet_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def inception_(input_shape=input_shape, units=1024, activation='relu', dropout=0.5):
    inception_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False,
                                                        input_shape=input_shape)
    inception_model.trainable = False
    model = tf.keras.Sequential()
    model.add(inception_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def densenet_(input_shape=input_shape, units=1024, activation='relu', dropout=0.5):
    densenet_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    densenet_model.trainable = False
    model = tf.keras.Sequential()
    model.add(densenet_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def mobilenet_(input_shape=input_shape, units=1024, activation='relu', dropout=0.5):
    mobilenet_model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    mobilenet_model.trainable = False
    model = tf.keras.Sequential()
    model.add(mobilenet_model)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # VGG model and results
    m1 = vgg_()
    r1 = train_test_model(m1)
    # ResNet model and results
    m2 = resnet_()
    r2 = train_test_model(m2)
    # Inception model and results
    m3 = inception_()
    r3 = train_test_model(m3)
    # DenseNet model and results
    m4 = densenet_()
    r4 = train_test_model(m4)
    # MobileNet
    m5 = mobilenet_()
    r5 = train_test_model(m5)
