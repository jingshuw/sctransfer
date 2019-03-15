import os, tempfile, shutil
import anndata
import numpy as np
import tensorflow as tf
import pandas as pd
from scipy.sparse import csr_matrix
import re
import pickle
import gc

from .io import read_dataset, normalize, write_text_matrix
from . import train_joint as tj
from . import network_joint as nj
import math



def autoencode(n_inoutnodes_human,
               n_inoutnodes_mouse,
               shared_size,
               adata = None,
               mtx_file=None,
               pred_adata=None,
               species = None,
               nonmissing_indicator=None,
               initial_file= "",
               out_dir=".",
               write_output_to_tsv = False,
               save_data = False,
               verbose = True, verbose_sum = True, verbose_fit = 1,
               batch_size = 32,
               data_name = '',
               net_kwargs={},
               training_kwargs={}): ###############

    if adata is None:
        if mtx_file is None:
            print('Either adata or mtx_file should be provided')
            return
        adata = anndata.read_mtx(mtx_file).transpose()
        if data_name == '':
            data_name = re.sub(r'.*/', '', mtx_file)
            data_name = data_name.replace('.mtx', '') + '_'

    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'

    ## add other information into AnnData instances
    adata.X = csr_matrix(adata.X)

    if species is not None:
        adata.uns['species'] = species
         
    adata.uns['data_type'] = 'UMI'

    # set seed for reproducibility
    np.random.seed(42)
    tf.set_random_seed(42)


 #   print(type(adata.X))
    adata = read_dataset(adata,
                         transpose=False,
                         test_split=False,
                         verbose = verbose,
                         copy=False)

    if 'X_dca' in adata.obsm_keys():
        filter_min_counts = False
        size_factors=False
        adata.X = csr_matrix(adata.obsm['X_dca'])
        if pred_adata:
            pred_adata.X = csr_matrix(pred_adata.obsm['X_dca'])
    else:
        filter_min_counts = True
        size_factors = True

    adata = normalize(adata,
                      filter_min_counts = filter_min_counts,
                      size_factors=size_factors,
                      logtrans_input=True)
    adata.uns['shared'] = adata.X.tocsc()[:, 0:shared_size].tocsr()

   # print(type(adata.X))

    if pred_adata:
        pred_adata.X = csr_matrix(pred_adata.X)
        pred_adata.uns['species'] = species
        pred_adata.uns['data_type'] = 'UMI'
        pred_adata = read_dataset(pred_adata,
                transpose=False, verbose = verbose,
                test_split=False)
        pred_adata = normalize(pred_adata,
                size_factors=size_factors,
                logtrans_input=True)
        pred_adata.uns['shared'] = pred_adata.X.tocsc()[:, 0:shared_size].tocsr()




 
    if nonmissing_indicator is None:
        nonmissing_indicator = 1


    net = nj.JointAutoencoder(input_size_human=n_inoutnodes_human,
                           input_size_mouse=n_inoutnodes_mouse,
                           shared_size = shared_size,
                           **net_kwargs)
   

    net.build()

    if (initial_file != ""):
        net.load_weights(initial_file)
        print("Weights loaded from %s!" % initial_file)

    model = tj.train_joint(adata[adata.obs.DCA_split == 'train'], 
            adata.uns['shared'],
            net, 
            output_dir=out_dir, batch_size = batch_size,
            save_weights = True, 
            verbose = verbose, verbose_sum = verbose_sum, verbose_fit = verbose_fit,
            nonmissing_indicator = nonmissing_indicator,
            **training_kwargs)

    model.load_weights("%s/weights.hdf5" % out_dir)



 
    if pred_adata:
        del adata
        res = net.predict(pred_adata, pred_adata.uns['shared'])
        del model,net
        gc.collect()
        pred_adata.obsm['X_dca'] = res['mean_norm']

        if write_output_to_tsv:
            print('Saving files ...')
            write_text_matrix(res['mean_norm'], 
                    os.path.join(out_dir, data_name + 'pred_mean_norm.tsv'))

        if save_data:
            with open(os.path.join(out_dir, data_name + 'pred_adata.pickle'), 'wb') as f:
                pickle.dump(pred_adata, f, protocol=4)
                f.close()

        return pred_adata


    res = net.predict(adata, adata.uns['shared'])
    del model,net
    gc.collect()

    adata.obsm['X_dca'] = res['mean_norm']


    if write_output_to_tsv:
        print('Saving files ...')
        write_text_matrix(res['mean_norm'], 
            os.path.join(out_dir, data_name + 'mean_norm.tsv'))
         #   write_text_matrix(res['dispersion'], 
         #       os.path.join(out_dir, data_name + 'dispersion.tsv'))
    if save_data:
        with open(os.path.join(out_dir, data_name + 'adata.pickle'), 'wb') as f:
            pickle.dump(adata, f, protocol=4)
            f.close()

 

    return adata
