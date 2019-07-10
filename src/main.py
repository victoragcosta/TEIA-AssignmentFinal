# Programa para implementar um classificador de gêneros musicais.

# Metadados:
PROGRAM_NAME = 'musiclassifier'
VERSION_NUM = '0.0.1'

# Bibliotecas:
import argparse
import keras
import numpy as np
import pandas as pd

# Módulos de usuário:
import cnn
import dlt

# Argumentos do programa:
parser = argparse.ArgumentParser(prog=f'{PROGRAM_NAME}',
                                 description='Musical genre classifier.')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help="batch size used for training the neural network.")
parser.add_argument('-e', '--epoch-num', default=5, type=int,
                    help="number of training epochs run by the program.")
parser.add_argument('-f', '--format', default='spectrogram',
                    choices=["chroma_stft", "spectrogram"],
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

# Data extraction:
print("Extracting data... ")

dlt_obj = dlt.DLT(format=args.format,
                  train_percent=args.train_percent,
                  sample_t_size=args.clip_length//10,
                  sample_split=args.split_num)

train_num = (args.track_num * args.train_percent)//100
test_num = (args.track_num * (100 - args.train_percent))//100

train_data = dlt_obj.get_train(
  n_audio=train_num,
)
test_data = dlt_obj.get_test(
  n_audio=test_num
)

print("Data extracted!")

# Train model example
# model = cnn.init(input_shape)
# cnn.train(model, x_train, y_train, x_test, y_test)
