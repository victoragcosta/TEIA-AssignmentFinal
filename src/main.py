# Programa para implementar um classificador de gêneros musicais.

# Número de versão:
PROGRAM_NAME = 'musiclassifier'
VERSION_NUM = '0.0.1'

# Bibliotecas
import numpy as np
import argparse
import pandas as pd
import keras

# Argumentos do programa:
parser = argparse.ArgumentParser(prog=f'{PROGRAM_NAME}',
                                 description='Musical genre classifier.')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    help="batch size used for training the neural network.")
parser.add_argument('-e', '--epoch-num', default=5, type=int,
                    help="number of training epochs run by the program.")
parser.add_argument('-f', '--format', default='wavelength',
                    choices=["wavelength", "spectrogram"],
                    help="music format used to train the neural network.")
parser.add_argument('-l', '--clip-length', default=20, type=int,
                    help="length of music training data (in TENTHS of second).")
parser.add_argument('-o', '--output-dir', default='out/',
                    help="output directory for the program's data plots.")
parser.add_argument('-v', '--version', action='version',
                    version=f'%(prog)s {VERSION_NUM}')
args = parser.parse_args()
