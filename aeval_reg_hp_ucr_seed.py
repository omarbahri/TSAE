#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:12:08 2022

@author: omar
"""

import os
import numpy as np
import sys
from ae import ae_val_bn_reg_hp_overallbest as ae
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import random
import csv
import fcntl
import pandas as pd
from tensorboard.plugins.hparams import api as hp

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def fit_transform_classifier(X_train, X_test, output_directory, nb_epochs,
                             hparams, batch_size, overall_best, weights_directory=None):
    
    if len(X_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension 
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    input_shape = X_train.shape[1:]
    
    print('hparams', hparams)
    print('batch_size', batch_size)
    
    model = ae.AE(output_directory, input_shape, hparams,
                  batch_size, patience=200, overall_best=overall_best, verbose=True, load_weights= False,
                  save_weights=2, save_logs=True)
    
    overall_best = model.fit(X_train, nb_epochs)
    
    X_train_pred = model.my_predict_last(X_train)
    X_test_pred = model.my_predict_last(X_test)
    return X_train_pred, X_test_pred, overall_best
    
data_path = os.path.join()
results_path = os.path.join()

name = sys.argv[1]
seed = int(sys.argv[2])
scale = int(sys.argv[3])
    
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

nb_epochs = 3000

HP_LR = hp.HParam('lr', hp.RealInterval(1e-8, 2.0))
HP_L2 = hp.HParam('l2', hp.RealInterval(1e-3, 1e0))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.4))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LOSS = hp.HParam('loss', hp.Discrete(['mse']))

# with tf.summary.create_file_writer(os.path.join(output_directory, 'logs', 'hparam_tuning')).as_default():
#   hp.hparams_config(
#     hparams=[HP_LR, HP_L2, HP_DROPOUT, HP_OPTIMIZER, HP_LOSS],
#     # metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
#   )

for name in [name]:  
    # try:
        print("Loaded Dataset.."+str(name))
        
        X_train, y_train = readucr(os.path.join(data_path, name, name + "_TRAIN.tsv"))
        X_test, y_test = readucr(os.path.join(data_path, name, name + "_TEST.tsv"))
        
        if scale==1:
            mean = np.mean(X_train)
            std = np.std(X_train)
            X_train = (X_train - mean) / std
            X_test = (X_test - mean) / std
            
            output_csv_path = os.path.join(results_path, 'orig_ucr_' + str(seed) +\
                   '_' + str(nb_epochs) + '_aeval_reg_scale_f1macro.csv')
            output_directory = os.path.join(results_path, name, 'orig', 'aeval_reg_scale')
                
        elif scale==2:
            min_val = np.min(X_train)
            max_val = np.max(X_train)
            
            if min_val >= 0:
                X_train = (X_train - min_val) / (max_val - min_val)
                X_test = (X_test - min_val) / (max_val - min_val)
                
                output_csv_path = os.path.join(results_path, 'orig_ucr_' + str(seed) +\
                       '_' + str(nb_epochs) + '_aeval_reg_minmax01_f1macro.csv')
                output_directory = os.path.join(results_path, name, 'orig', 'aeval_reg_minmax01')
            
            else:
                X_train = (X_train - min_val) / (max_val - min_val) * 2 - 1
                X_test = (X_test - min_val) / (max_val - min_val) * 2 - 1
                
                output_csv_path = os.path.join(results_path, 'orig_ucr_' + str(seed) +\
                       '_' + str(nb_epochs) + '_aeval_reg_minmax-11_f1macro.csv')
                output_directory = os.path.join(results_path, name, 'orig', 'aeval_reg_minmax-11')
         
        elif scale==3:
            min_val = np.min(X_train)
            max_val = np.max(X_train)
            X_train = (X_train - min_val) / (max_val - min_val)
            X_test = (X_test - min_val) / (max_val - min_val)
            
            output_csv_path = os.path.join(results_path, 'orig_ucr_' + str(seed) +\
                   '_' + str(nb_epochs) + '_aeval_reg_minmax01_f1macro.csv')
            output_directory = os.path.join(results_path, name, 'orig', 'aeval_reg_minmax01')
        
        else:
            output_csv_path = os.path.join(results_path, 'orig_ucr_' + str(seed) +\
                   '_' + str(nb_epochs) + '_aeval_reg_f1macro.csv')
            output_directory = os.path.join(results_path, name, 'orig', 'aeval_reg')
        
        try:    
            output_csv = pd.read_csv(output_csv_path, header=None)
        except Exception:
            print('Gonna create this file later')
        
        # if not os.path.exists(output_directory):
        #     os.makedirs(output_directory)
        
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)
        
        if len(X_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            
        p = np.random.RandomState(seed=42).permutation(len(y_train))
        X_train, y_train = X_train[p], y_train[p]
        
        # * 0.8 because 0.2 is used for validation
        # if int(X_train.shape[0] * 0.8) % 8 == 0:
        #     batch_size = 8
        # elif int(X_train.shape[0] * 0.8) % 4 == 0:
        #     batch_size = 4
        # elif int(X_train.shape[0] * 0.8) % 5 == 0:
        #     batch_size = 5
        # elif int(X_train.shape[0] * 0.8) % 2 == 0:
        #     batch_size = 2
        # else:
        #     print(int(X_train.shape[0] * 0.8))
        #     print('Change batch size')
        
        batch_size = 8
        overall_best = np.inf
             
        for optimizer in HP_OPTIMIZER.domain.values:
            for loss in HP_LOSS.domain.values:
                for _ in range(100):
                
                    # Sample random values for each hyperparameter
                    lr = random.uniform(HP_LR.domain.min_value, HP_LR.domain.max_value)
                    l2 = random.uniform(HP_L2.domain.min_value, HP_L2.domain.max_value)
                    l2 = 0
                    dropout = random.uniform(HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value)
                    
                    hparams = {'lr': lr, 'l2': l2, 'dropout': dropout,
                               'optimizer': optimizer, 'loss': loss}
                    
                    name_csv = '_'.join([name, str(hparams['lr']), 
                                        str(hparams['l2']), str(hparams['dropout']),
                                        str(hparams['optimizer']), str(hparams['loss'])])                     
                        
                    try:
                        if name_csv in list(output_csv[output_csv.columns[0]]):
                            print('Already Done')
                            continue
                    except Exception:
                        print('')
                    
                    X_train_pred, X_test_pred, overall_best = fit_transform_classifier(X_train, X_test,
                                         output_directory, nb_epochs, hparams, batch_size, overall_best)
                   
                    print('overall_best:', str(overall_best))
                    
                    train_mse_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
            
                    # Get reconstruction loss threshold.
                    threshold = np.max(train_mse_loss)
                    print("Reconstruction error threshold: ", threshold)
             
                    # Get test mse loss.
                    test_mse_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
                    test_mse_loss = test_mse_loss.reshape((-1))
             
                    # Detect all the samples which are anomalies.
                    anomalies = test_mse_loss > threshold
                    print("Number of anomaly samples: ", np.sum(anomalies))
                    print("Indices of anomaly samples: ", np.where(anomalies))
                     
                    with open(output_csv_path, 'a', newline='') as fw:
                        fcntl.flock(fw, fcntl.LOCK_EX)
                        writer = csv.writer(fw)
                        writer.writerow([name_csv, np.mean(train_mse_loss), np.mean(test_mse_loss), np.sum(anomalies)])
                        fcntl.flock(fw, fcntl.LOCK_UN)
                        
    # except Exception as ex:
    #     print(ex)
        
