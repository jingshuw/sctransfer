import os

from .loss import NB, ZINB

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.optimizers as opt
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import Iterator
from scipy.sparse import csr_matrix


def train_joint(adata, 
                shared_mat, 
                network, 
                output_dir=None, 
                nonmissing_indicator=None,
                optimizer='rmsprop', 
                learning_rate=None, 
                train_on_full=False,
                epochs=300, 
                reduce_lr=4,         
                early_stop=6, 
                batch_size=32, 
                clip_grad=5., 
                save_weights=False,
                tensorboard=False, 
                verbose_sum=True, verbose=True, verbose_fit = 1, **kwargs):

    # reshape to include an extra node for UMI/nonUMI indicator node
    shared_mat = csr_matrix((shared_mat.data, shared_mat.indices, 
        shared_mat.indptr), shape = (adata.n_obs, network.shared_size + 1))

    ## add the indicator node for UMI/nonUMI
    if adata.uns['data_type'] == 'nonUMI':
        temp = csr_matrix((np.array([1] * adata.n_obs), 
            np.array([network.shared_size] * adata.n_obs),
            np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, network.shared_size + 1))
        shared_mat = temp + shared_mat



    if adata.uns['species'] == 'Human':
        mouse_mat = csr_matrix((adata.n_obs, network.input_size_mouse + 1), dtype = np.int8)
        human_mat = csr_matrix((adata.X.data, adata.X.indices, adata.X.indptr), 
            shape = (adata.n_obs, adata.n_vars + 1))
        if adata.uns['data_type'] == 'UMI':
            model = network.model['hn_umi']
            loss = NB(model.get_layer('dispersion_human_umi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                debug = network.debug).loss
        elif adata.uns['data_type'] == 'nonUMI':
            model = network.model['hn_nonumi']
            temp = csr_matrix((np.array([1] * adata.n_obs), 
                np.array([adata.n_vars] * adata.n_obs),
                np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, adata.n_vars + 1))
            human_mat = temp + human_mat
            loss = ZINB(model.get_layer('pi_human').output, 
                    theta = model.get_layer('dispersion_human_nonumi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                    ridge_lambda = network.ridge, debug = network.debug).loss



    if adata.uns['species'] == 'Mouse':
        human_mat = csr_matrix((adata.n_obs, network.input_size_human + 1), dtype = np.int8)
        mouse_mat = csr_matrix((adata.X.data, adata.X.indices, adata.X.indptr), 
            shape = (adata.n_obs, adata.n_vars + 1))
        if adata.uns['data_type'] == 'UMI':
            model = network.model['ms_umi']
            loss = NB(model.get_layer('dispersion_mouse_umi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                    debug = network.debug).loss
        elif adata.uns['data_type'] == 'nonUMI':
            model = network.model['ms_nonumi']
            temp = csr_matrix((np.array([1] * adata.n_obs), 
                np.array([adata.n_vars] * adata.n_obs),
                np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, adata.n_vars + 1))
            mouse_mat = temp + mouse_mat
            loss = ZINB(model.get_layer('pi_mouse').output, 
                    theta = model.get_layer('dispersion_mouse_nonumi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                ridge_lambda = network.ridge, debug = network.debug).loss




    
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

    inputs = {'human': human_mat.toarray(), 
            'mouse': mouse_mat.toarray(),
            'shared': shared_mat.toarray(),
            'size_factors': adata.obs.size_factors}


    if tensorboard:
        tb_log_dir = os.path.join(output_dir, 'tb')
        tb_cb = TensorBoard(log_dir=tb_log_dir,
                histogram_freq = 0, batch_size = 32, write_graph = False, 
                embeddings_freq = 5, embeddings_data = inputs)
        callbacks.pop()
        callbacks.append(tb_cb)

    if verbose_sum:
        model.summary()

    inputs = {'human': human_mat, 
            'mouse': mouse_mat,
            'shared': shared_mat,
            'size_factors': adata.obs.size_factors}

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

    return model

def evaluate_joint(adata, shared_mat, network, optimizer='rmsprop',
        batch_size = 32, verbose = 1, 
        nonmissing_indicator=None):

    shared_mat = csr_matrix((shared_mat.data, shared_mat.indices, 
        shared_mat.indptr), shape = (adata.n_obs, network.shared_size + 1))

    if adata.uns['data_type'] == 'nonUMI':
        temp = csr_matrix((np.array([1] * adata.n_obs), 
            np.array([network.shared_size] * adata.n_obs),
            np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, network.shared_size + 1))
        shared_mat = temp + shared_mat



    if adata.uns['species'] == 'Human':
        mouse_mat = csr_matrix((adata.n_obs, network.input_size_mouse + 1), dtype = np.int8)
        human_mat = csr_matrix((adata.X.data, adata.X.indices, adata.X.indptr), 
            shape = (adata.n_obs, adata.n_vars + 1))
        if adata.uns['data_type'] == 'UMI':
            model = network.model['hn_umi']
            loss = NB(model.get_layer('dispersion_human_umi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                debug = network.debug).loss
        elif adata.uns['data_type'] == 'nonUMI':
            model = network.model['hn_nonumi']
            temp = csr_matrix((np.array([1] * adata.n_obs), 
                np.array([adata.n_vars] * adata.n_obs),
                np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, adata.n_vars + 1))
            human_mat = temp + human_mat
            loss = ZINB(model.get_layer('pi_human').output, 
                    theta = model.get_layer('dispersion_human_nonumi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator, 
                ridge_lambda = network.ridge, debug = network.debug).loss



    if adata.uns['species'] == 'Mouse':
        human_mat = csr_matrix((adata.n_obs, network.input_size_human + 1), dtype = np.int8)
        mouse_mat = csr_matrix((adata.X.data, adata.X.indices, adata.X.indptr), 
            shape = (adata.n_obs, adata.n_vars + 1))
        if adata.uns['data_type'] == 'UMI':
            model = network.model['ms_umi']
            loss = NB(model.get_layer('dispersion_mouse_umi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                    debug = network.debug).loss
        elif adata.uns['data_type'] == 'nonUMI':
            model = network.model['ms_nonumi']
            temp = csr_matrix((np.array([1] * adata.n_obs), 
                np.array([adata.n_vars] * adata.n_obs),
                np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, adata.n_vars + 1))
            mouse_mat = temp + mouse_mat
            loss = ZINB(model.get_layer('pi_mouse').output, 
                    theta = model.get_layer('dispersion_mouse_nonumi').theta_exp, 
                    nonmissing_indicator = nonmissing_indicator,
                ridge_lambda = network.ridge, debug = network.debug).loss



 

    model.compile(loss=loss, optimizer = optimizer)
    
    err = model.evaluate(x = {'mouse': mouse_mat, 'human': human_mat,
        'shared': shared_mat,
        'size_factors': adata.obs.size_factors},
        verbose = verbose,
        y = adata.raw.X, batch_size = batch_size)

    return err



