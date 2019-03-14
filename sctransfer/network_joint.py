import os

import numpy as np
import scanpy.api as sc

import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.objectives import mean_squared_error
from keras import backend as K

import tensorflow as tf

from .layers import ConstantDispersionLayer, SliceLayer, ColWiseMultLayer, ElementwiseDense
from scipy.sparse import csr_matrix



MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


class JointAutoencoder():
    def __init__(self,
                 input_size_human,
                 input_size_mouse,
                 shared_size,
                 hidden_size =(128, 64, 32, 64, 128),
                 ridge=0.,
                 hidden_dropout=0.,
                 input_dropout=0.,
                 batchnorm=True,
                 activation='relu',
                 init='glorot_uniform',
                 nonmissing_indicator=None,
                 debug = False):


        self.input_size_human = input_size_human
        self.input_size_mouse = input_size_mouse
        self.shared_size = shared_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.model = {}
        self.input_layer_mouse = None
        self.input_layer_human = None
        self.input_layer_shared = None
        self.sf_layer = None
        self.debug = debug
        self.decoder_output_mouse = None
        self.decoder_output_human = None
        self.ridge = ridge
        self.output_size_human = input_size_human
        self.output_size_mouse = input_size_mouse


        self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)


    def build(self):

        ## build input layer, add one node for UMI/non-UMI indicator
        self.input_layer_mouse = Input(shape=(self.input_size_mouse + 1,), name='mouse')
        self.input_layer_human = Input(shape = (self.input_size_human + 1,), 
                name = 'human')
        self.input_layer_shared = Input(shape = (self.shared_size + 1,), name = 'shared')
     
        self.sf_layer = Input(shape=(1,), name='size_factors')
        
        last_hidden_mouse = self.input_layer_mouse
        last_hidden_human = self.input_layer_human
        last_hidden_joint = self.input_layer_shared
        if self.input_dropout > 0.0:
            last_hidden_mouse = Dropout(self.input_dropout, name='mouse_input_drop')(last_hidden_mouse)
            last_hidden_human = Dropout(self.input_dropout, name='human_input_drop')(last_hidden_human)
            last_hidden_joint = Dropout(self.input_dropout, name='joint_input_drop')(last_hidden_joint)

        ## increase mouse hidden layer
        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'ms_center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'ms_enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'ms_dec%s' % (i-center_idx)
                stage = 'decoder'
   
            last_hidden_mouse = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                name=layer_name)(last_hidden_mouse)
            if self.batchnorm:
                last_hidden_mouse = BatchNormalization(center=True, scale=False)(last_hidden_mouse)

            last_hidden_mouse = Activation(self.activation, name='%s_act'%layer_name)(last_hidden_mouse)

            if hid_drop > 0.0:
                last_hidden_mouse = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden_mouse)

        ## increase human hidden layer
        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'hn_center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'hn_enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'hn_dec%s' % (i-center_idx)
                stage = 'decoder'
   
            last_hidden_human = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                name=layer_name)(last_hidden_human)
            if self.batchnorm:
                last_hidden_human = BatchNormalization(center=True, scale=False)(last_hidden_human)

            last_hidden_human = Activation(self.activation, name='%s_act'%layer_name)(last_hidden_human)
            if hid_drop > 0.0:
                last_hidden_human = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden_human)

        ## increase shared hidden layer
        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'jt_center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'jt_enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'jt_dec%s' % (i-center_idx)
                stage = 'decoder'
   
            last_hidden_joint = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                name=layer_name)(last_hidden_joint)
            if self.batchnorm:
                last_hidden_joint = BatchNormalization(center=True, scale=False)(last_hidden_joint)

            last_hidden_joint = Activation(self.activation, name='%s_act'%layer_name)(last_hidden_joint)
            if hid_drop > 0.0:
                last_hidden_joint = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden_joint)

        
        

        self.decoder_output_human = keras.layers.concatenate([last_hidden_human, last_hidden_joint], 
                name = "human_out")
        self.decoder_output_mouse = keras.layers.concatenate([last_hidden_mouse, last_hidden_joint], 
                name = "mouse_out")


        self.build_output()

    def build_output(self):
        mean_no_act_mouse =  Dense(self.output_size_mouse, activation=None, kernel_initializer=self.init,
               name='mean_mouse_no_act')(self.decoder_output_mouse)
        mean_no_act_human =  Dense(self.output_size_human, activation=None, kernel_initializer=self.init,
                name='mean_human_no_act')(self.decoder_output_human)

        mean_mouse = Activation(MeanAct, name = 'mean_mouse')(mean_no_act_mouse)
        mean_human = Activation(MeanAct, name = 'mean_human')(mean_no_act_human)


        self.model['joint_meannorm'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse,
            self.input_layer_shared], outputs = [mean_mouse, mean_human])

        ## add zero inflation layer for nonUMI data
        pi_mouse = ElementwiseDense(self.output_size_mouse, activation='sigmoid', kernel_initializer=self.init,
                   name='pi_mouse')(mean_no_act_mouse)
        pi_human = ElementwiseDense(self.output_size_human, activation='sigmoid', kernel_initializer=self.init,
                   name='pi_human')(mean_no_act_human)



        # Plug in dispersion parameters via fake dispersion layer
        ## allow different dispersion parameters for UMI and nonUMI
        disp_human_umi = ConstantDispersionLayer(name='dispersion_human_umi')
        disp_human_nonumi = ConstantDispersionLayer(name='dispersion_human_nonumi')
        mean_human_umi = disp_human_umi(mean_human)
        mean_human_nonumi = disp_human_nonumi(mean_human)

        output_human_umi = ColWiseMultLayer(name='output_human_umi')([mean_human_umi, self.sf_layer])
        output_human_nonumi = ColWiseMultLayer(name='output_human_nonumi')([mean_human_nonumi, self.sf_layer])
        output_human_nonumi = SliceLayer(0, name='slice_output_human_nonumi')([output_human_nonumi, pi_human])
        
        self.model['hn_umi'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse,
            self.input_layer_shared, self.sf_layer], 
                outputs = output_human_umi)
        self.model['hn_nonumi'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse, 
            self.input_layer_shared, self.sf_layer], outputs = output_human_nonumi)
        self.model['hn_meannorm'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse, 
            self.input_layer_shared], outputs = mean_human)

        disp_mouse_umi = ConstantDispersionLayer(name='dispersion_mouse_umi')
        disp_mouse_nonumi = ConstantDispersionLayer(name='dispersion_mouse_nonumi')
        mean_mouse_umi = disp_mouse_umi(mean_mouse)
        mean_mouse_nonumi = disp_mouse_nonumi(mean_mouse)

        output_mouse_umi = ColWiseMultLayer(name='output_mouse_umi')([mean_mouse_umi, self.sf_layer])
        output_mouse_nonumi = ColWiseMultLayer(name='output_mouse_nonumi')([mean_mouse_nonumi, self.sf_layer])
        output_mouse_nonumi = SliceLayer(0, name='slice_output_mouse_nonumi')([output_mouse_nonumi, pi_mouse])
        
        self.model['ms_umi'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse, 
            self.input_layer_shared, self.sf_layer], 
                outputs = output_mouse_umi)
        self.model['ms_nonumi'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse, 
            self.input_layer_shared, self.sf_layer], outputs = output_mouse_nonumi)
        self.model['ms_meannorm'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse,
            self.input_layer_shared], outputs = mean_mouse)
        self.model['joint_output'] = Model(inputs=[self.input_layer_human, self.input_layer_mouse,
            self.input_layer_shared, self.sf_layer],
                outputs = [output_mouse_umi, output_mouse_nonumi, 
                    output_human_umi, output_human_nonumi])
       


    
    def load_weights(self, filename):
        self.model['joint_output'].load_weights(filename)



    def predict(self, adata, shared_mat, colnames = None):

        res = {}
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values
        
        shared_mat = csr_matrix((shared_mat.data, shared_mat.indices, 
            shared_mat.indptr), shape = (adata.n_obs, self.shared_size + 1))

     #   if adata.uns['data_type'] == 'nonUMI':
     #       temp = np.ones(adata.n_obs)
     #   else:
     #       temp = np.zeros(adata.n_obs)
     #
     #   shared_mat = np.column_stack((shared.mat, temp))


        ## add the indicator node for nonUMI data
        if adata.uns['data_type'] == 'nonUMI':
            temp = csr_matrix((np.array([1] * adata.n_obs), 
                np.array([adata.n_vars] * adata.n_obs),
                np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, adata.n_vars + 1))
            temp_shared = csr_matrix((np.array([1] * adata.n_obs), 
                np.array([self.shared_size] * adata.n_obs),
                np.array(range(adata.n_obs + 1))), shape = (adata.n_obs, self.shared_size + 1))
        else:
            temp = csr_matrix((adata.n_obs, adata.n_vars + 1), dtype = np.int8)
            temp_shared = csr_matrix((adata.n_obs, self.shared_size + 1), dtype = np.int8)


        shared_mat = shared_mat + temp_shared


        if adata.uns['species'] == 'Human':
            mouse_mat = csr_matrix((adata.n_obs, self.input_size_mouse + 1), dtype = np.int8)
            human_mat = csr_matrix((adata.X.data, adata.X.indices, adata.X.indptr), 
                shape = (adata.n_obs, adata.n_vars + 1)) + temp

          #  human_mat = np.column_stack((adata.X.data, temp))

            res['mean_norm'] = self.model['hn_meannorm'].predict({'mouse': mouse_mat, 'human': human_mat,
                'shared':shared_mat})

        if adata.uns['species'] == 'Mouse':
            human_mat = csr_matrix((adata.n_obs, self.input_size_human + 1), dtype = np.int8)
            mouse_mat = csr_matrix((adata.X.data, adata.X.indices, adata.X.indptr), 
                shape = (adata.n_obs, adata.n_vars + 1)) + temp

       #     mouse_mat = np.column_stack((adata.X.data, temp))


            res['mean_norm'] = self.model['ms_meannorm'].predict({'mouse': mouse_mat, 'human': human_mat,
                'shared':shared_mat})

        return res






