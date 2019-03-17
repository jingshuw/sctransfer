## modified from the dca package

import numpy as np
import scipy as sp
import pandas as pd
import scanpy.api as sc
from sklearn.model_selection import train_test_split


### Split the dataset, make sure input is count
def read_dataset(adata, transpose=False, test_split=False, copy=False, verbose = True):

    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError

#    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    type_error = 'Make sure that the dataset is of one of the two types: UMI, nonUMI(RPKM/TPM)'
    assert adata.uns['data_type'] in ['UMI', 'nonUMI'], type_error


#    if adata.X.size < 50e6 and adata.uns['data_type'] != 'nonUMI': # check if adata.X is integer only if array is small
#        if sp.sparse.issparse(adata.X):
#            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
#        else:
#            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if transpose: adata = adata.transpose()

    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)
        spl.iloc[test_idx] = 'test'
        adata.obs['DCA_split'] = spl.values
        adata.uns['train_idx'] = train_idx
        adata.uns['test_idx'] = test_idx
    else:
        adata.obs['DCA_split'] = 'train'
    
    adata.obs['DCA_split'] = adata.obs['DCA_split'].astype('category')
    if verbose:
        print('### Autoencoder: Successfully preprocessed {} genes and {} cells.'.format(adata.n_vars, 
            adata.n_obs), flush = True)

    return adata

## filter all zero cells, lib_size normalization, log-transformation 
def normalize(adata, filter_min_counts=True, size_factors=True, logtrans_input=True):

    if filter_min_counts:
     #   sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)

    if adata.raw is None:
        if size_factors or logtrans_input:
            adata.raw = adata.copy()
        else:
            adata.raw = adata


    if adata.uns['data_type'] != 'nonUMI':
        n_counts = adata.obs.n_counts
    else:
        n_counts=adata.X.sum(axis = 1)
   
    if size_factors:
        sc.pp.normalize_per_cell(adata, counts_per_cell_after = 10000)
        adata.obs['size_factors'] = n_counts / 10000 #/ np.median(adata.obs.n_counts)

    if logtrans_input:
        sc.pp.log1p(adata)



    return adata

def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.3f')

