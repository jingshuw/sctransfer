from .sctransfer.saverx_train import SaverXTrain
import anndata
import numpy as np
import os 
from glob import glob
import sys
sys.stdout.flush()


def pretrainFolder(folder,
        species_list,
        data_type_list = None,
        out_dir = ".", 
        initial_file = "",
        n_mouse = 21122,
        n_human = 21183,
        n_shared = 15494,
        batch_size = 100,
        pretrain_kwargs = {}):

    mtx_files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.mtx'))]
    nonmissing_files = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*nonmissing.txt'))]
    if data_type_list is None:
        data_type_list = ['UMI'] * len(mtx_files)
    if len(species_list) == 1:
        species_list = species_list * len(mtx_files)

    idx = np.arange(len(mtx_files))
    np.random.seed(42)
    np.random.shuffle(idx)

    nonmissing_indicator_list = []

    for f in nonmissing_files:
        nonmissing_indicator_list.append(np.loadtxt(f))

    data_list = []
    for ff in mtx_files:
        print(ff)
        data_list.append(anndata.read_mtx(ff).transpose())
    print(species_list)
    print(data_type_list)

    for i in range(len(mtx_files)):
        data_list[i].uns['species'] = species_list[i]
        print(species_list[i])
        data_list[i].uns['data_type'] = data_type_list[i]
        print(data_type_list[i])


    result = SaverXTrain(data_list, n_human, n_mouse, n_shared,
        out_dir = out_dir, 
        nonmissing_indicator_list = nonmissing_indicator_list,
        initial_file = initial_file, 
        batch_size = batch_size,
        **pretrain_kwargs)


