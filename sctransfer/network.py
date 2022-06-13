## code simplified from the dca package

import os
import numpy as np
import scanpy as sc

import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.objectives import mean_squared_error
from keras import backend as K

import tensorflow as tf

from .loss import NB
from .layers import ConstantDispersionLayer, ColWiseMultLayer


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)


class Autoencoder():
    def __init__(self,
                 input_size,
                 output_size=None,
                 hidden_size=(64, 32, 64),
                 hidden_dropout=0.,
                 input_dropout=0.,
                 batchnorm=True,
                 activation='relu',
                 init='glorot_uniform',
                 nonmissing_indicator = None,
                 debug = False):

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.input_dropout = input_dropout
        self.batchnorm = batchnorm
        self.activation = activation
        self.init = init
        self.loss = None
        self.extra_models = {}
        self.model = None
        self.input_layer = None
        self.sf_layer = None
        self.debug = debug
        self.nonmissing_indicator = nonmissing_indicator

        if self.output_size is None:
            self.output_size = input_size


        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name='count')
        self.sf_layer = Input(shape=(1,), name='size_factors')
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name='input_dropout')(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'dec%s' % (i-center_idx)
                stage = 'decoder'


            last_hidden = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                name=layer_name)(last_hidden)
     
            if self.batchnorm:
                last_hidden = BatchNormalization(center=True, scale=False)(last_hidden)
            ### TODO: check why scale = False

            last_hidden = Activation(self.activation, name='%s_act'%layer_name)(last_hidden)
            if hid_drop > 0.0:
                last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

        self.decoder_output = last_hidden
        self.build_output()

    def build_output(self):

        ## For Gaussian loss
        self.loss = mean_squared_error
        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                     name='mean')(self.decoder_output)
        output = ColWiseMultLayer(name='output')([mean, self.sf_layer])

        # keep unscaled output as an extra model
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)


        ######## ADD WEIGHTS ###########


    def load_weights(self, filename):
        self.model.load_weights(filename)


    def predict(self, adata, colnames=None, dimreduce=True, reconstruct=True, error=True):

        res = {}
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

   #     print('Calculating reconstructions...')

        res['mean_norm'] = self.extra_models['mean_norm'].predict(adata.X)
        
        return res


class NBConstantDispAutoencoder(Autoencoder):

    def build_output(self):
        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                     name='mean')(self.decoder_output)

        # Plug in dispersion parameters via fake dispersion layer
        disp = ConstantDispersionLayer(name='dispersion')
        mean = disp(mean)

        output = ColWiseMultLayer(name='output')([mean, self.sf_layer])

        nb = NB(disp.theta_exp, nonmissing_indicator = self.nonmissing_indicator)
        self.extra_models['dispersion'] = lambda :K.function([], [nb.theta])([])[0].squeeze()
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)


    def predict(self, adata, colnames=None, **kwargs):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values
        res = super().predict(adata, colnames=colnames, **kwargs)

        res['dispersion'] = self.extra_models['dispersion']()
  
        return res

