import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import sys

from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Softmax, BatchNormalization, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from prepare import data,targets, algo_list
from config import EPOCHS, BATCH_SIZE, INPUT_SHAPE


ACT = tf.nn.leaky_relu
CLASSES = len(algo_list)
INPUT_SHAPE = data.shape

def parseArguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Training Parser")
    parser.add_argument("-epochs", type=int, default=EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("-batch_size", type=int, default=BATCH_SIZE,
                        help="Set the training batch size.")
    # parser.add_argument("-model", type = str, default=MODEL_ARCH,
    #                     help = "Provide a model architecture")
    args = parser.parse_args(args)
    return args


def build_model(input_shape = INPUT_SHAPE):
    model = Sequential([
        Dense(128,activation=ACT, input_shape = input_shape),
#        BatchNormalization(),
        Dropout(0.2),
        #Dense(64, activation=ACT),
#        BatchNormalization(),
        #Dropout(0.2),
        #Dense(16, activation=ACT),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='sgd',loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# create a tf data object
def plot(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # ACCURACY
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # LOSS
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

model = build_model()
args = parseArguments()

def train_model(data, target,model = model, args = args):
    early_stopper = EarlyStopping(monitor = 'val_loss',patience=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy')
    cbs = [reduce_lr]

    history = model.fit(data,target, validation_split = 0.15, epochs = args.epochs, batch_size=args.batch_size,shuffle=True, callbacks=cbs)
    return history

history = train_model(data,targets)
plot(history)



