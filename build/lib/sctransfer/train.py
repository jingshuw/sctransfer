## simplified code from the dca package

import os

from .loss import NB
#from hyper import hyper

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator


def train(adata, 
          network, 
          output_dir=None, 
          optimizer='rmsprop', 
          learning_rate=None, 
          train_on_full=False,
          epochs=300, 
          reduce_lr=4, 
          early_stop=6, 
          batch_size=32, 
          clip_grad=5., 
          save_weights=False,
          nonmissing_indicator = None,
          tensorboard=False, 
          verbose=True, verbose_sum = True, verbose_fit = 1, **kwargs):


    
    model = network.model

    loss = NB(model.get_layer('dispersion').theta_exp,
                nonmissing_indicator = nonmissing_indicator,
                debug = network.debug).loss


    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)

    model.compile(loss=loss, optimizer=optimizer)

    # Callbacks
    callbacks = []

    if save_weights and output_dir is not None:
        checkpointer = ModelCheckpoint(filepath="%s/weights.hdf5" % output_dir,
                                       verbose=verbose,
                                       save_weights_only=True,
                                       save_best_only=True)
        callbacks.append(checkpointer)
    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)
    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir, histogram_freq=1, write_grads=True)
        callbacks.append(tb_cb)

    if verbose_sum:
        model.summary()

    inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}

    output = adata.raw.X

    if train_on_full:
        validation_split = 0
    else:
        validation_split = 0.1

    loss = model.fit(inputs, output,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks,
                     validation_split=validation_split,
                     verbose=verbose_fit,
                     **kwargs)
    # https://github.com/tensorflow/tensorflow/issues/3388
    # K.clear_session()

    return loss

def evaluate(adata, network, optimizer='rmsprop',
        batch_size = 32,
        nonmissing_indicator=None):

    model = network.model

    loss = NB(model.get_layer('dispersion').theta_exp,
              nonmissing_indicator = nonmissing_indicator,
                debug = network.debug).loss
  
    model.compile(loss=loss, optimizer = optimizer)
    
    err = model.evaluate(x = {'count': adata.X,
        'size_factors': adata.obs.size_factors},
        y = adata.raw.X, 
        batch_size = batch_size)

    return err


