## SAVER-X without pretraining
## code simplified from the dca package

import os, tempfile, shutil
import anndata
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import re, gc
import sys
sys.stdout.flush()

from .io import read_dataset, normalize, write_text_matrix
from .train import train
from .network import NBConstantDispAutoencoder




def autoencode(adata = None,
               mtx_file = None,
               pred_adata=None, ## cross-validation purpose
               pred_mtx_file = None,
               out_dir=".",
               write_output_to_tsv = False,
               save_data = False,
               verbose = True, verbose_sum = True, verbose_fit = 1, 
               batch_size = 32,
               data_name = "",
               nonmissing_indicator = None,
               net_kwargs={}): ###############

    if adata is None:
        if mtx_file is None:
            print('Either adata or mtx_file should be provided')
            return
        adata = anndata.read_mtx(mtx_file).transpose()
        if data_name == "":
            data_name = re.sub(r'.*/', '', mtx_file)
            data_name = data_name.replace('.mtx', '') + '_'

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'

    adata.uns['data_type'] = 'UMI'
 

    # set seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)


    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         verbose = verbose,
                         copy=False)

    adata = normalize(adata,
                      filter_min_counts = True,
                      size_factors=True,
                      logtrans_input=True)

    if pred_adata or pred_mtx_file:
        if pred_adata is None:
            pred_adata = anndata.read_mtx(pred_mtx_file).transpose()
        pred_adata.uns['data_type'] = 'UMI'
        pred_adata = read_dataset(pred_adata,
                transpose=False,
                test_split=False, 
                verbose = verbose,
                copy=False)
        pred_adata = normalize(pred_adata,
                size_factors=True,
                logtrans_input=True)
 

    net = NBConstantDispAutoencoder(input_size=adata.n_vars,
            nonmissing_indicator = nonmissing_indicator,
            **net_kwargs)
    net.build()


    loss = train(adata[adata.obs.DCA_split == 'train'], 
            net, 
            output_dir=out_dir, 
            batch_size = batch_size,
            save_weights = True, 
            nonmissing_indicator = nonmissing_indicator,
            verbose = verbose, verbose_sum = verbose_sum, verbose_fit = verbose_fit)

    net.load_weights("%s/weights.hdf5" % out_dir)
    
    if pred_adata or pred_mtx_file:
        del adata
        res = net.predict(pred_adata)
        pred_adata.obsm['X_dca'] = res['mean_norm']
        del net,loss
        gc.collect()

        if write_output_to_tsv:
            print('Saving files ...')
            write_text_matrix(res['mean_norm'], 
                    os.path.join(out_dir, data_name + 'pred_mean_norm.tsv'))

        if save_data:
            with open(os.path.join(out_dir, data_name + 'pred_adata.pickle'), 'wb') as f:
                pickle.dump(pred_adata, f, protocol=4)
                f.close()
        return pred_adata


    res = net.predict(adata)
    adata.obsm['X_dca'] = res['mean_norm']
    adata.var['X_dca_dispersion'] = res['dispersion']

    if write_output_to_tsv:
        print('Saving files ...')
        write_text_matrix(res['mean_norm'], 
                    os.path.join(out_dir, data_name + 'mean_norm.tsv'))
        write_text_matrix(res['dispersion'], 
                    os.path.join(out_dir, data_name + 'dispersion.tsv'))
    if save_data:
            with open(os.path.join(out_dir, data_name + 'adata.pickle'), 'wb') as f:
                pickle.dump(adata, f, protocol=4)
                f.close()

    del net,loss
    gc.collect()


    return adata
