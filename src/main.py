# Programa para implementar um classificador de gêneros musicais.

# Metadados:
PROGRAM_NAME = 'musiclassifier'
VERSION_NUM = '0.0.1'

# Faz o parse primeiro, depois carrega. O programa responde a -h e -v mais rápido
import argparse

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
parser.add_argument('--input-prefix', default="res", type=str,
                    dest='inputprefix',
                    help="input prefix for loading models.")
parser.add_argument('-f', '--format', default='spectrogram',
                    choices=["chroma_stft", "spectrogram", "melspectrogram", "mfcc"],
                    help="music format used to train the neural network.")
parser.add_argument('-l', '--clip-length', default=10, type=int,
                    dest='clip_length',
                    help="length of music training data (in TENTHS of second).")
parser.add_argument('-n', '--track-number', default=1000, type=int,
                    dest='track_num',
                    help="number of tracks used for training.")
parser.add_argument('-o', '--output-dir', default='out/',
                    help="output directory for the program's data plots.")
parser.add_argument('-s', '--split-num', default=1, type=int,
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
parser.add_argument('-L', '--load',
                    dest='should_load', action='store_true',
                    help="if it is present, it will load a model based on "+
                    "the --input-prefix name given. If not present, it will "+
                    "create a new one.")
parser.add_argument('-p', '--padding',
                    dest='enable_padding', action='store_true',
                    help="enable padding, if not set it will remove examples that doesn't fit first example shape.")
parser.add_argument('--data-batch', default=100, type=int,
                    dest='data_batch',
                    help="size of each data batch to load.")
parser.set_defaults(should_load=False)
parser.set_defaults(enable_padding=False)
args = parser.parse_args()

# Bibliotecas:
import keras
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd

# Módulos de usuário:
import cnn
import dlt
import csvconverter
import plot_tools

## Main module functions
def load_data_batch(dlt_obj, batch_size):
    print("Extracting data batch... ")
    train_num = (batch_size * args.train_percent)//100
    test_num = (batch_size * (100 - args.train_percent))//100
    
    train_data = dlt_obj.get_train(n_audio=train_num)
    if train_data is None: raise Exception("No more data!")
    
    test_data = dlt_obj.get_test(n_audio=test_num)
    if test_data is None: raise Exception("No more data!")
    
    # Add padding or remove examples with different shape
    first_shape = train_data[0][1][0].shape
    if args.enable_padding:
        dlt.extend_data(train_data, first_shape)
        dlt.extend_data(test_data, first_shape)
    else:
        train_data = [x for x in train_data if x[1][0].shape == first_shape]
        test_data = [x for x in test_data if x[1][0].shape == first_shape]
    
    ## Extract features with normalization:
    train_features = np.array([x[1][0]/800 for x in train_data])
    test_features = np.array([x[1][0]/800 for x in test_data])
    
    ## Extract and format outputs:
    y_train = np.array([x[0] for x in train_data])
    y_test = np.array([x[0] for x in test_data])
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
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
        
    print("Data batch extracted... ")
        
    return x_train, y_train, x_test, y_test, input_shape


def save_model_info(model, training_stats):
    ## Save training history in .csv file
    csvconverter.savecsv(csvconverter.converter(training_stats),
                         'results/' + args.outputprefix+'_trainhistory.csv')
    
    ## Save model
    model.save('results/' + args.outputprefix+'_model.h5')

def save_batch_info(model, x_train, y_train, x_test, y_test, batch_num):
    ## Save predicted
    y_train_pred = model.predict(x_train)
    y_train_pred = list(map(lambda x: np.argmax(x), y_train_pred))
    
    ## Get train labels
    y_train_labels = list(map(lambda x: np.argmax(x), y_train))
    
    plot_tools.plot_confusion_matrix(y_train_labels, y_train_pred, plot=False, save_image=True, image_path='results/', image_name=args.outputprefix+'_train_confusion_matrix_' + str(batch_num) + '.png')
    
    ## Save validation predicted
    y_test_pred = model.predict(x_test)
    y_test_pred = list(map(lambda x: np.argmax(x), y_test_pred))
    
    ## Get validation labels
    y_test_labels = list(map(lambda x: np.argmax(x), y_test))
    
    plot_tools.plot_confusion_matrix(y_test_labels, y_test_pred, plot=False, save_image=True, image_path='results/', image_name=args.outputprefix+'_validation_confusion_matrix_' + str(batch_num) + '.png')
    plot_tools.get_and_plot_metrics(y_test_labels, y_test_pred, save_table=True, table_format='latex', file_path='results/', file_name=args.outputprefix+'_validation_metrics_' + str(batch_num))

def load_dlt():
    return dlt.DLT(format=args.format,
                  train_percent=args.train_percent,
                  sample_t_size=args.clip_length//10,
                  sample_split=args.split_num)

## Training settings:
num_classes = 10

# Data loader initialization
dlt_obj = load_dlt()

# Train initial model:
if args.should_load:
    model = load_model('results/' + args.inputprefix+'_model.h5')
else:
    model = None

## Train statistics
training_stats = {'val_loss': [], 'val_acc': [], 'loss': [], 'acc': []}

# Train while examples found
batch_num = 0
num_epochs = args.epoch_num
while True:
    # Ask if it should stop
    if batch_num*args.data_batch >= args.track_num:
        num_epochs = int(input("Quantas épocas deseja ainda fazer? "))
        if num_epochs == 0:
            break
        else:
            batch_num = 0
            dlt_obj = load_dlt()
    
    try:
        x_train, y_train, x_test, y_test, input_shape = load_data_batch(dlt_obj, args.data_batch)
    except:
        break
    
    # Create model if first time
    if model is None:
        print("Creating CNN model for the first time... ")
        model = cnn.init(
            input_shape,
            cnn_format=[
                {'n_filters': 6, 'window_size' : (4,4), 'pool_size': (2,2), 'dropout': 0.25},
                {'n_filters': 6, 'window_size' : (4,4), 'pool_size': (2,2), 'dropout': 0.25}
            ]
        )
    
    print("Training data epoch... ")
    history = cnn.train(model, x_train, y_train, x_test, y_test, epochs=num_epochs)
    
    ## Append new training stats
    for key in history.history.keys():
        training_stats[key] = training_stats[key] + history.history[key]
    
    print("Saving information... ")
    save_batch_info(model, x_train, y_train, x_test, y_test, batch_num)
    
    batch_num += 1
    
print("Saving model and stats... ")
save_model_info(model, training_stats)


