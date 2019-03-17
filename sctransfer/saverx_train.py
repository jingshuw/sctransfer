import os, tempfile, shutil
import anndata
import numpy as np
import tensorflow as tf
from keras import backend as K
import pandas as pd
from scipy.sparse import csr_matrix

from . import io as io
from . import train_joint as tj
from . import network_joint as nj
import math
import gc



def SaverXTrain(adata_list,
        n_inoutnodes_human,
        n_inoutnodes_mouse,
        shared_size,
        nonmissing_indicator_list=None,
        test_split=True,
        initial_file= "",
        out_dir=".",
        batch_size = 32,
        epochs = 300,
        reduce_lr = 4,
        early_stop = 6,
        net_kwargs={},
        training_kwargs={}): ###############


    # set seed for reproducibility
    np.random.seed(42)
    tf.set_random_seed(42)



    net = nj.JointAutoencoder(input_size_human= n_inoutnodes_human,
                           input_size_mouse= n_inoutnodes_mouse,
                           shared_size = shared_size,
                           **net_kwargs)

    net.build()
    net.model['joint_output'].summary()

    if (initial_file != ""):
        net.load_weights(initial_file)
        print("Weights loaded from %s!" % initial_file)



############ split the data into train and test #######################

    print("Number of datasets is %s" % len(adata_list), flush = True)
    for i in range(len(adata_list)):
        adata_list[i] = io.read_dataset(adata_list[i],
                transpose=False,
                test_split=test_split,
                copy=False)
        if 'X_dca' in adata_list[i].obsm_keys():
            filter_min_counts = False
            size_factors = False
            adata_list[i].X = csr_matrix(adata_list[i].obsm['X_dca'])
        else:
            filter_min_counts = True
            size_factors = True

        adata_list[i] = io.normalize(adata_list[i],
                filter_min_counts = filter_min_counts,
                size_factors=size_factors,
                logtrans_input=True)
        temp = adata_list[i].X.tocsc()[:, 0:net.shared_size].tocsr()
        if test_split:
            adata_list[i].uns['shared_train'] = temp[adata_list[i].uns['train_idx'], :] 
            adata_list[i].uns['shared_test'] = temp[adata_list[i].uns['test_idx'], :]
        else:
            adata_list[i].uns['shared_train'] = temp
    if nonmissing_indicator_list is None:
        nonmissing_indicator_list = [1] * len(adata_list)

    print("Data preprossessed ...")

    gc.collect()


  #### loop across all datasets for training ############
   

    eval_err = math.inf
    stop_count = 0
    learning_rate = None
    old_lr = 0.001
    for k in range(epochs):  
        idx = list(range(len(adata_list)))
        np.random.shuffle(idx)
        adata_list = [adata_list[i] for i in idx]
        nonmissing_indicator_list = [nonmissing_indicator_list[i] for i in idx]
        print("Calculating the %s/%s epoch ..." % (k, epochs), flush = True)
        for i in range(len(adata_list)):
             model =tj.train_joint(adata_list[i][adata_list[i].obs.DCA_split == 'train'], 
                    adata_list[i].uns['shared_train'],
                    net, 
                    output_dir=out_dir, batch_size = batch_size,
                    save_weights = False,
                    learning_rate = learning_rate,
                    reduce_lr = None, early_stop = None,
                    verbose_sum = False,
                    verbose = False,
                    verbose_fit = 0,
                    nonmissing_indicator = nonmissing_indicator_list[i],
                    epochs = 1, train_on_full = True,
                    **training_kwargs)
        err = 0
        n_test = 0
        for i in range(len(adata_list)):
            n = adata_list[i][adata_list[i].obs.DCA_split == "test"].n_obs
            err += tj.evaluate_joint(adata_list[i][adata_list[i].obs.DCA_split == "test"], 
                    adata_list[i].uns['shared_test'],
                    net, batch_size = batch_size, 
                    verbose = 0,
                    nonmissing_indicator = nonmissing_indicator_list[i]) * n
            n_test += n

        err = err/n_test
        print("Number of test samples is %s --------------------------->>>> evaluation loss: %.4f" % (n_test, err), flush = True)
        if err > eval_err:
            stop_count += 1
            print("Evaluation error not improved from %.4f" % eval_err)
        else:
            stop_count = 0
            eval_err = err
            print("Evaluation error reduced. Save model weights to file %s/weights.hdf5" % out_dir)
            net.model['joint_output'].save_weights("%s/weights.hdf5" % out_dir)

        if stop_count == reduce_lr:
            learning_rate = old_lr * 0.1
            print("Learning rate reduced from %.4f to %.4f" % (old_lr, learning_rate), flush = True)
            old_lr = learning_rate


        if stop_count == early_stop:
            print("Training stops by early stopping as the validation loss is not improved", flush = True)
            break

        gc.collect()

    return None
