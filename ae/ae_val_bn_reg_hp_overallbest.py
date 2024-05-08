import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import regularizers
# import tensorflow_addons as tfa
import numpy as np
import time
import shutil

import matplotlib
# from utils.utils import save_test_duration

# matplotlib.use('agg'%M)
import matplotlib.pyplot as plt

# from utils.utils import save_logs
# from utils.utils import calculate_metrics

from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp

import os 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, *args, initial_value_threshold=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_val_loss = initial_value_threshold
        self.last_best_epoch = None
        # self.model_saved = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.last_best_epoch = epoch
        # if self.model.saved:
        #     self.model_saved = True
        super().on_epoch_end(epoch, logs)
        
    def on_train_end(self, logs=None):
        if self.last_best_epoch is None:
            print("No model saved during training.")
        else:
            print("Best val_loss:", self.best_val_loss, "at epoch", self.last_best_epoch)
        super().on_train_end(logs)


class AE:
    def __init__(self, output_directory, input_shape, hparams, batch_size=12, patience=50, 
             overall_best=np.inf, verbose=False, build=True, load_weights=False, save_weights=2, 
             save_logs=True, weights_directory=''):
        params_dir = '_'.join([str(hparams['lr']), 
                        str(hparams['l2']), str(hparams['dropout']),
                        str(hparams['optimizer']), str(hparams['loss'])])
        self.overall_directory = os.path.join(output_directory, 'overall')
        if not os.path.exists(self.overall_directory) and save_weights==2:
            os.makedirs(self.overall_directory)
        self.output_directory = os.path.join(output_directory, params_dir)
        if not os.path.exists(self.output_directory) and save_weights==1:
            os.makedirs(self.output_directory)
        self.log_dir = os.path.join(output_directory, 'logs', 'hparams', params_dir)
        self.weights_directory = weights_directory
        self.lr = hparams['lr']
        self.l2 = hparams['l2']
        self.dropout = hparams['dropout']
        self.optimizer = hparams['optimizer']
        self.loss = hparams['loss']
        self.hparams = hparams
        self.patience = patience
        self.overall_best = overall_best
        self.batch_size = batch_size
        self.save_weights = save_weights
        self.save_logs = save_logs
        self.params_dir = params_dir
        
        print(self.lr, self.l2, self.dropout, self.optimizer, self.loss)
        
        if build == True:
            self.model = self.build_model(input_shape)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(os.path.join(self.weights_directory, 'last_model.hdf5'))
            elif save_weights==1:
                self.model.save_weights(os.path.join(self.output_directory, 'model_init.hdf5'))
        return
        
    def build_encoder(self, input_shape):
        input_layer = keras.layers.Input(shape=input_shape)

        # Conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=5, strides=1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=self.dropout)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)

        # Conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=11, strides=1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=self.dropout)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)

        # Conv block -3
        conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        conv3 = keras.layers.Dropout(rate=self.dropout)(conv3)

        # Split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :256])(conv3)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 256:])(conv3)

        # Attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])

        # Flatten
        flatten = keras.layers.Flatten()(multiply_layer)

        self.encoder = keras.Model(inputs=input_layer, outputs=flatten)
        return self.encoder

    def build_decoder(self, encoded_shape, input_shape):
        encoded_input = keras.layers.Input(encoded_shape)

        # Reshape to the shape after the last max pooling layer in the encoder
        reshape = keras.layers.Reshape((-1, 256))(encoded_input)

        # Deconv block -1
        deconv1 = keras.layers.Conv1DTranspose(filters=256, kernel_size=21, strides=1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))(reshape)
        deconv1 = tf.keras.layers.BatchNormalization()(deconv1)
        deconv1 = keras.layers.PReLU(shared_axes=[1])(deconv1)
        deconv1 = keras.layers.Dropout(rate=self.dropout)(deconv1)
        deconv1 = keras.layers.UpSampling1D(size=2)(deconv1)

        # Deconv block -2
        deconv2 = keras.layers.Conv1DTranspose(filters=128, kernel_size=11, strides=1, padding='same',
                                    kernel_regularizer=regularizers.l2(self.l2))(deconv1)
        deconv2 = tf.keras.layers.BatchNormalization()(deconv2)
        deconv2 = keras.layers.PReLU(shared_axes=[1])(deconv2)
        deconv2 = keras.layers.Dropout(rate=self.dropout)(deconv2)
        deconv2 = keras.layers.UpSampling1D(size=2)(deconv2)
            
        # Deconv block -3
        decoded_output = keras.layers.Conv1DTranspose(filters=input_shape[1], kernel_size=5, strides=1, padding='same',
                                                      activation='sigmoid',
                                                      kernel_regularizer=regularizers.l2(self.l2))(deconv2)
        
        # Restitute original shape if not divisible by 4 (2 maxpooling layers)
        if decoded_output.shape[1] != input_shape[0]:
            decoded_output = keras.layers.Conv1DTranspose(filters=input_shape[1], kernel_size=5, strides=1, padding='valid',
                                                          activation='sigmoid',
                                                          kernel_regularizer=regularizers.l2(self.l2))(deconv2)
            ks = decoded_output.shape[1] - input_shape[0] + 1
            decoded_output = keras.layers.Conv1D(filters=input_shape[1], kernel_size=ks, strides=1, padding='valid',
                                        kernel_regularizer=regularizers.l2(self.l2))(decoded_output)
        
        self.decoder = keras.Model(inputs=encoded_input, outputs=decoded_output)
        return self.decoder

    def build_model(self, input_shape):
        encoder = self.build_encoder(input_shape)
        decoder = self.build_decoder(encoder.output_shape[1], input_shape)
        autoencoder_output = decoder(encoder.output)
        model = keras.Model(inputs=encoder.input, outputs=autoencoder_output)
        
        if self.optimizer == 'adam':
            opt = keras.optimizers.Adam(self.lr)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(self.lr)
        elif self.optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(self.lr)
        
        model.compile(loss=self.loss, optimizer=opt,
                      metrics=[self.loss])

        file_path = os.path.join(self.output_directory, 'best_model.hdf5')
        
        self.callbacks = [tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss", patience=self.patience, mode="min")]
        
        if self.save_weights==1:
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, 
                                       monitor='val_loss', save_best_only=True,
                                       )
            self.callbacks = self.callbacks + [model_checkpoint]
            
        elif self.save_weights==2:
            custom_checkpoint = CustomModelCheckpoint(
                filepath=os.path.join(self.overall_directory, 'maybe_best_model.hdf5'),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                mode='min',
                initial_value_threshold=self.overall_best
            )
            
            self.callbacks = self.callbacks + [custom_checkpoint]


        if self.save_logs:
            tensorboard_callback = TensorBoard(log_dir=self.log_dir)
            hp_keras_callback = hp.KerasCallback(self.log_dir, self.hparams)
            
            self.callbacks = self.callbacks + [tensorboard_callback] + [hp_keras_callback]

        return model
    
    def fit(self, x_train, nb_epochs=100):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        # batch_size = 12
        # nb_epochs = 1500

        mini_batch_size = int(min(x_train.shape[0] / 10, self.batch_size))

        start_time = time.time()
        
        hist = self.model.fit(x_train, x_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, 
                              validation_split=0.2,  callbacks=self.callbacks)

        duration = time.time() - start_time
        
        if self.save_weights==1:
            self.model.save(os.path.join(self.output_directory, 'last_model.hdf5'))
        
        if self.save_weights==2:
            best_val_loss = self.callbacks[1].best_val_loss
            
            if best_val_loss < self.overall_best:
                self.overall_best = best_val_loss
                with open(os.path.join(self.overall_directory, 'last_best.txt'), 'a') as file:
                    file.write(self.params_dir + ': ' + str(self.overall_best) + '\n')
                self.model.save(os.path.join(self.overall_directory, 'last_model.hdf5'))
                shutil.copy(os.path.join(self.overall_directory, 'maybe_best_model.hdf5'), 
                            os.path.join(self.overall_directory, 'best_model.hdf5'))
                
        keras.backend.clear_session()

        #return df_metrics
        return self.overall_best
    
    def my_predict(self, x_test):
        model_path = os.path.join(self.output_directory, 'best_model.hdf5')
        model = keras.models.load_model(model_path)
        X_pred = model.predict(x_test)
        return X_pred
    
    def my_predict_last(self, x_test):
        X_pred = self.model.predict(x_test)
        return X_pred
