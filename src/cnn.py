import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten, Dropout

def init(input_shape, 
         cnn_format=[
           {'n_filters': 4, 'window_size' : (4,4), 'pool_size': (2,2), 'dropout': 0.25},
           {'n_filters': 4, 'window_size' : (4,4), 'pool_size': (2,2), 'dropout': 0.25}
         ], 
         mlp_hidden_layers=[200], 
         output_size = 10):
  
  model = Sequential()
  
  model.add(Conv2D(cnn_format[0]['n_filters'], cnn_format[0]['window_size'], padding="same", input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=cnn_format[0]['pool_size']))
  model.add(Dropout(cnn_format[0]['dropout']))
  
  for f in cnn_format[1:]:
    model.add(Conv2D(f['n_filters'], f['window_size'], padding="same"))
    model.add(MaxPooling2D(pool_size=f['pool_size']))
    model.add(Dropout(f['dropout']))
  
  model.add(Flatten())
  
  for hl_size in mlp_hidden_layers:
    model.add(Dense(hl_size))
  
  model.add(Dense(output_size))
  model.add(Activation("softmax"))
  model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])
  
  return model

def train(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=100):
  return model.fit(x_train, y_train,
                   batch_size=batch_size,
                   epochs=epochs,
                   verbose=1,
                   validation_data=(x_test, y_test))


