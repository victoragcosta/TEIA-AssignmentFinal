# Programa para implementar um classificador de gêneros musicais.

# Metadados:
PROGRAM_NAME = 'musiclassifier'
VERSION_NUM = '0.0.1'

# Bibliotecas:
import argparse
import keras
from keras import backend as K
import numpy as np
import pandas as pd

# Módulos de usuário:
import cnn
import dlt
import csvconverter
import plot_tools

# Argumentos do programa:
parser = argparse.ArgumentParser(prog=f'{PROGRAM_NAME}',
                                 description='Musical genre classifier.')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help="batch size used for training the neural network.")
parser.add_argument('-e', '--epoch-num', default=10, type=int,
                    dest='epoch_num',
                    help="number of training epochs run by the program.")
parser.add_argument('--output-prefix', default="res", type=str,
                    dest='outputprefix',
                    help="output of train history as CSV file.")
parser.add_argument('-f', '--format', default='spectrogram',
                    choices=["chroma_stft", "spectrogram", "melspectrogram", "mfcc"],
                    help="music format used to train the neural network.")
parser.add_argument('-l', '--clip-length', default=100, type=int,
                    dest='clip_length',
                    help="length of music training data (in TENTHS of second).")
parser.add_argument('-n', '--track-number', default=1000, type=int,
                    dest='track_num',
                    help="number of tracks used for training.")
parser.add_argument('-o', '--output-dir', default='out/',
                    help="output directory for the program's data plots.")
parser.add_argument('-s', '--split-num', default=5, type=int,
                    dest='split_num',
                    help="number of splits made when classifying a single " +
                    "clip (each split is a fraction of the clip used in a " +
                    "majority vote).")
parser.add_argument('-t', '--train-percent', default=70, type=int,
                    dest='train_percent',
                    help="percentage of audios used in training (the rest is " +
                    "used in tests).")
parser.add_argument('-v', '--version', action='version',
                    version=f'%(prog)s {VERSION_NUM}')
args = parser.parse_args()

## Training settings:
num_classes = 10

# Data extraction:
print("Extracting data... ")

dlt_obj = dlt.DLT(format=args.format,
                  train_percent=args.train_percent,
                  sample_t_size=args.clip_length//10,
                  sample_split=args.split_num)

train_num = (args.track_num * args.train_percent)//100
test_num = (args.track_num * (100 - args.train_percent))//100

train_data = dlt_obj.get_train(n_audio=train_num)
test_data = dlt_obj.get_test(n_audio=test_num)

# Remove different input shapes
first_shape = train_data[0][1][0].shape
train_data = [x for x in train_data if x[1][0].shape == first_shape]
test_data = [x for x in test_data if x[1][0].shape == first_shape]

## Extract features with normalization:
train_features = np.array([x[1][0]/800 for x in train_data])
test_features = np.array([x[1][0]/800 for x in test_data])

## Extract and format outputs:
train_labels = np.array([x[0] for x in train_data])
test_labels = np.array([x[0] for x in test_data])

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

## Format backend data:

img_rows = len(train_features[0])
img_cols = len(train_features[0][0])

if K.image_data_format() == 'channels_first':
    x_train = train_features.reshape(len(train_features), 1, img_rows, img_cols)
    x_test = test_features.reshape(len(test_features), 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = train_features.reshape(len(train_features), img_rows, img_cols, 1)
    x_test = test_features.reshape(len(test_features), img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print("Data extracted!")

# Train model:

model = cnn.init(input_shape)
history = cnn.train(model, x_train, train_labels, x_test, test_labels, epochs=args.epoch_num)

## Save training history in .csv file
csvconverter.savecsv(csvconverter.converter(history.history),
                     'results/' + args.outputprefix+'_trainhistory.csv')

## Save model
model.save('results/' + args.outputprefix+'_model.h5')

## Save predicted
y_train_pred = model.predict(x_train)
y_train_pred = list(map(lambda x: np.argmax(x), y_train_pred))

## Get train labels
y_train_labels = list(map(lambda x: np.argmax(x), train_labels))

plot_tools.plot_confusion_matrix(y_train_labels, y_train_pred, save_image=True, image_path='results/', image_name=args.outputprefix+'_train_confusion_matrix.png')

## Save validation predicted
y_test_pred = model.predict(x_test)
y_test_pred = list(map(lambda x: np.argmax(x), y_test_pred))

## Get validation labels
y_test_labels = list(map(lambda x: np.argmax(x), test_labels))

plot_tools.plot_confusion_matrix(y_test_labels, y_test_pred, save_image=True, image_path='results/', image_name=args.outputprefix+'_validation_confusion_matrix.png')
plot_tools.get_and_plot_metrics(y_test_labels, y_test_pred, save_table=True, table_format='latex', file_path='results/', file_name=args.outputprefix+'_validation_metrics')

