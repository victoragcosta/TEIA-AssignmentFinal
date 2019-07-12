# Programa para implementar um classificador de gêneros musicais.

# Faz o parse primeiro, depois carrega. O programa responde a -h e -v mais rápido
import argparse

parser = argparse.ArgumentParser(description='Passe as opções -n, --data-batch, -f, -l e -p usadas no treinamento. Esse módulo precisa dessas informações para separar treinamento de validação da mesma forma que foi feito no treinamento. Passe também --model o modelo que vai usar para predizer os valores.')
parser.add_argument('--prefix',     type=str, dest='prefix', default='res')
parser.add_argument('-n',           type=int, dest='n', required=True)
parser.add_argument('--data-batch', type=int, dest='dataBatch', required=True)
parser.add_argument('--xmargin',    type=int, dest='xmargin', default=2)
parser.add_argument('--ymargin',    type=int, dest='ymargin', default=2)
parser.add_argument('--model',      type=str, dest='model', required=True)
parser.add_argument('-f', '--format', type=str, dest='format', required=True)
parser.add_argument('-t',           type=int, dest='train_percent', default=70)
parser.add_argument('-l',           type=int, dest='clip_length', required=True)
parser.add_argument('-s',           type=int, dest='split_num', default=1)
parser.add_argument('-p', '--padding', dest='enable_padding', action='store_true')

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

# Functions

num_classes = 10

def get_max_shape(data1,data2):
    return (
            np.array([x[1][0].shape[0] for x in data1]+[x[1][0].shape[0] for x in data2]).max()+args.xmargin,
            np.array([x[1][0].shape[1] for x in data1]+[x[1][0].shape[1] for x in data2]).max()+args.ymargin
           )

def load_dlt():
    return dlt.DLT(format=args.format,
                  train_percent=args.train_percent,
                  sample_t_size=args.clip_length//10,
                  sample_split=args.split_num)


def load_data_batch(dlt_obj, batch_size, globalShape):
    train_num = (batch_size * args.train_percent)//100
    test_num = (batch_size * (100 - args.train_percent))//100
    
    print("Extracting data batch: max " + str(train_num) + " for train and max " + str(test_num) + " for test...")
    
    train_data = dlt_obj.get_train(n_audio=train_num)
    if train_data is None: raise IndexError("No more data!")
    
    test_data = dlt_obj.get_test(n_audio=test_num)
    if test_data is None: raise IndexError("No more data!")
    
    # Add padding or remove examples with different shape
    
    if args.enable_padding:
        if globalShape is None:
            globalShape = get_max_shape(train_data,test_data)
        dlt.extend_data(train_data, globalShape)
        dlt.extend_data(test_data, globalShape)
        
    else: 
        if globalShape is None:
            globalShape = train_data[0][1][0].shape
    
    train_data = [x for x in train_data if x[1][0].shape == globalShape]
    test_data = [x for x in test_data if x[1][0].shape == globalShape]
    
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
        
    return x_train, y_train, x_test, y_test, input_shape, globalShape



# Data loader initialization
dlt_obj = load_dlt()

# Load model
model = load_model(args.model)

# Obtain stats by data batches
batch_num = 0
globalShape = None
ytrue_train = []
ytrue_test = []
ypred_train = None
ypred_test = None
while batch_num*args.dataBatch < args.n:
    
    try:
        x_train, y_train, x_test, y_test, input_shape, globalShape = load_data_batch(dlt_obj, args.dataBatch, globalShape)
    except IndexError:
        break
    
    print("Predicting with model... ")
    
    # Predict train
    ypred_train_tmp = list(map(lambda x: np.argmax(x), model.predict(x_train)))
    if ypred_train is None: ypred_train = ypred_train_tmp
    else: ypred_train = np.concatenate([ypred_train, ypred_train_tmp])
    
    ytrue_train += list(map(lambda x: np.argmax(x), y_train))
    
    # Predict validation
    ypred_test_tmp = list(map(lambda x: np.argmax(x), model.predict(x_test)))
    if ypred_test is None: ypred_test = ypred_test_tmp
    else: ypred_test = np.concatenate([ypred_test, ypred_test_tmp])
    
    ytrue_test += list(map(lambda x: np.argmax(x), y_test))
    
    
    batch_num += 1

plot_tools.plot_confusion_matrix(ytrue_train, ypred_train, plot=False, save_image=True, image_path='results/', image_name=args.prefix+'_global_train_confusion_matrix.png')
plot_tools.plot_confusion_matrix(ytrue_test, ypred_test, plot=False, save_image=True, image_path='results/', image_name=args.prefix+'_global_validation_confusion_matrix.png')
plot_tools.get_and_plot_metrics(ytrue_test, ypred_test, save_table=True, table_format='latex', file_path='results/', file_name=args.prefix+'_global_validation_metrics')

print("Acurácia global de treinamento: " + str((ytrue_train == ypred_train).sum()/len(ytrue_train)))
print("Acurácia global de validação: " + str((ytrue_test == ypred_test).sum()/len(ytrue_test)))
