# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:23:13 2018

@author: Wendong Zheng
"""

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers.recurrent import Layer, RNN, GRU, GRUCell, LSTM, LSTMCell
from keras.layers.recurrent import _generate_dropout_mask, _generate_dropout_ones
from keras.legacy import interfaces

class GRUCell_Custom(GRUCell):
    def __init__(self, units, sigma=0.5, use_layer_norm=True, zoneout=0., 
                 center=True, scale=True,
                 negative_update_bias=False,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 **kwargs):
        super().__init__(units, **kwargs)
        self.sigma = sigma
        self.center = center
        self.scale = scale
        self.zoneout = zoneout
        self.use_layer_norm = use_layer_norm
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.negative_update_bias = negative_update_bias
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        if self.scale:
            self.gamma = self.add_weight(shape=(self.units * 3,),
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         trainable=True)

            self.gamma_z = self.gamma[: self.units]
            self.gamma_r = self.gamma[self.units: self.units * 2]
            self.gamma_h = self.gamma[self.units * 2:]

        if self.center:
            self.beta = self.add_weight(shape=(self.units * 3,),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        trainable=True)

            self.beta_z = self.beta[: self.units]
            self.beta_r = self.beta[self.units: self.units * 2]
            self.beta_h = self.beta[self.units * 2:]
            
        self.kernel = self.add_weight(shape=(input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.negative_update_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 3,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                                        self.units:
                                                        self.units * 2]
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            self.bias_z = self.bias[:self.units]
            self.bias_r = self.bias[self.units: self.units * 2]
            self.bias_h = self.bias[self.units * 2:]
        else:
            self.bias_z = None
            self.bias_r = None
            self.bias_h = None
        self.built = True
        
    def ln(self, x):
        if self.use_layer_norm:
            mean = K.mean(x, -1, keepdims=True)
            var = K.var(x, -1, keepdims=True)
            x_normed = (x - mean) / K.sqrt(var + self.sigma ** 2)
            return x_normed
        else:
            return x
        

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory

        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=3)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=3)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        if self.implementation == 1:
            if 0. < self.dropout < 1.:
                inputs_z = inputs * dp_mask[0]
                inputs_r = inputs * dp_mask[1]
                inputs_h = inputs * dp_mask[2]
            else:
                inputs_z = inputs
                inputs_r = inputs
                inputs_h = inputs
                
            x_z = K.dot(inputs_z, self.kernel_z)
            x_r = K.dot(inputs_r, self.kernel_r)
            x_h = K.dot(inputs_h, self.kernel_h)
            if self.use_bias:
                x_z = K.bias_add(x_z, self.bias_z)
                x_r = K.bias_add(x_r, self.bias_r)
                x_h = K.bias_add(x_h, self.bias_h)

            if 0. < self.recurrent_dropout < 1.:
                h_tm1_z = h_tm1 * rec_dp_mask[0]
                h_tm1_r = h_tm1 * rec_dp_mask[1]
                h_tm1_h = h_tm1 * rec_dp_mask[2]
            else:
                h_tm1_z = h_tm1
                h_tm1_r = h_tm1
                h_tm1_h = h_tm1
            
            recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel_z)
            recurrent_r = K.dot(h_tm1_r, self.recurrent_kernel_r)
            
            a_z = self.ln(x_z + recurrent_z)
            a_r = self.ln(x_r + recurrent_r)
            if self.scale:
                a_z *= self.gamma_z
                a_r *= self.gamma_r
            if self.center:
                a_z += self.beta_z
                a_r += self.beta_r
            z = self.recurrent_activation(a_z)
            r = self.recurrent_activation(a_r)
            
            
            recurrent_h = K.dot(r * h_tm1_h, self.recurrent_kernel_h)
            a_h = self.ln(x_h + recurrent_h)
            if self.scale:
                a_h *= self.gamma_h
            if self.center:
                a_h += self.beta_h
            hh = self.activation(a_h)
            
        # ignore implementation 2
        
        h = z * h_tm1 + (1 - z) * hh
        
        if 0 < self.dropout + self.recurrent_dropout + self.zoneout:
            if training is None:
                h._uses_learning_phase = True
                
        if 0 < self.zoneout < 1:
            h = K.in_train_phase(K.dropout(h - h_tm1, self.zoneout),
                                 h - h_tm1)
            h = h * (1. - self.zoneout) + h_tm1
        
        return h, [h]
        
class GRU_Custom(GRU):
    @interfaces.legacy_recurrent_support
    def __init__(self, units, sigma=0.5, use_layer_norm=True, zoneout=0.,
                 center=True, scale=True,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 negative_update_bias=False,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 reset_after=False,
                 **kwargs):

        cell = GRUCell_Custom(units, sigma=sigma, center=center, scale=scale, zoneout=zoneout,
                              use_layer_norm=use_layer_norm,
                              negative_update_bias=negative_update_bias,
                              activation=activation, 
                              recurrent_activation=recurrent_activation, 
                              use_bias=use_bias, 
                              kernel_initializer=kernel_initializer, 
                              recurrent_initializer=recurrent_initializer, 
                              bias_initializer=bias_initializer, 
                              gamma_initializer=gamma_initializer,
                              beta_initializer=beta_initializer,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              implementation=implementation)
        super(GRU, self).__init__(cell,
                                  return_sequences=return_sequences,
                                  return_state=return_state,
                                  go_backwards=go_backwards,
                                  stateful=stateful,
                                  unroll=unroll,
                                  **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        
        
class LSTMCell_Custom(LSTMCell):
    def __init__(self, units, sigma=0.5, use_layer_norm=True, zoneout_c=0., zoneout_h=0.,
                 **kwargs):
        super(LSTMCell_Custom, self).__init__(units, **kwargs)
        self.sigma = sigma
        self.use_layer_norm = use_layer_norm
        self.zoneout_c = zoneout_c
        self.zoneout_h = zoneout_h

    def ln(self, x):
        if self.use_layer_norm:
            mean = K.mean(x, -1, keepdims=True)
            var = K.var(x, -1, keepdims=True)
            x_normed = (x - mean) / K.sqrt(var + self.sigma ** 2)
            return x_normed
        else:
            return x
        
    def call(self, inputs, states, training=None):
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, K.shape(inputs)[-1]),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.recurrent_dropout,
                training=training,
                count=4)
        if (0 < self.zoneout_c < 1 and
                self._zoneout_mask_c is None):
            self._zoneout_mask_c = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.zoneout_c,
                training=training,
                count=1)
            
        if (0 < self.zoneout_h < 1 and
                self._zoneout_mask_h is None):
            self._zoneout_mask_h = _generate_dropout_mask(
                _generate_dropout_ones(inputs, self.units),
                self.zoneout_h,
                training=training,
                count=1)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            x_i = K.dot(inputs_i, self.kernel_i)
            x_f = K.dot(inputs_f, self.kernel_f)
            x_c = K.dot(inputs_c, self.kernel_c)
            x_o = K.dot(inputs_o, self.kernel_o)
            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
                
            i = self.recurrent_activation(self.ln(x_i + K.dot(h_tm1_i,
                                                              self.recurrent_kernel_i)))
            f = self.recurrent_activation(self.ln(x_f + K.dot(h_tm1_f,
                                                              self.recurrent_kernel_f)))
            c = f * c_tm1 + i * self.activation(self.ln(x_c + K.dot(h_tm1_c,
                                                                    self.recurrent_kernel_c)))
            o = self.recurrent_activation(self.ln(x_o + K.dot(h_tm1_o,
                                                              self.recurrent_kernel_o)))

        h = o * self.activation(self.ln(c))
        
        if 0 < self.dropout + self.recurrent_dropout + self.zoneout_c + self.zoneout_h:
            if training is None:
                h._uses_learning_phase = True
                
        if 0 < self.zoneout_h < 1:
            h = K.in_train_phase(K.dropout(h - h_tm1, self.zoneout_h),
                                 h - h_tm1)
            h = h * (1. - self.zoneout_h) + h_tm1
            
        if 0 < self.zoneout_c < 1:
            c = K.in_train_phase(K.dropout(c - c_tm1, self.zoneout_c),
                                 c - c_tm1)
            c = c * (1. - self.zoneout_c) + c_tm1
        
        return h, [h, c]



class LSTM_Custom(LSTM):
    @interfaces.legacy_recurrent_support
    def __init__(self, units, sigma=0.5, use_layer_norm=True, zoneout_c=0., zoneout_h=0., 
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = LSTMCell_Custom(units, sigma=sigma, use_layer_norm=use_layer_norm,
                               zoneout_c=zoneout_c, zoneout_h=zoneout_h,
                               activation=activation,
                               recurrent_activation=recurrent_activation,
                               use_bias=use_bias,
                               kernel_initializer=kernel_initializer,
                               recurrent_initializer=recurrent_initializer,
                               unit_forget_bias=unit_forget_bias,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=kernel_regularizer,
                               recurrent_regularizer=recurrent_regularizer,
                               bias_regularizer=bias_regularizer,
                               kernel_constraint=kernel_constraint,
                               recurrent_constraint=recurrent_constraint,
                               bias_constraint=bias_constraint,
                               dropout=dropout,
                               recurrent_dropout=recurrent_dropout,
                               implementation=implementation)
        super(LSTM, self).__init__(cell,
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        self.cell._zoneout_mask_h = None
        self.cell._zoneout_mask_c = None
        return super(LSTM, self).call(inputs,
                                      mask=mask,
                                      training=training,
                                      initial_state=initial_state)
    
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        #self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
    #print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        #return input_shape[0], input_shape[-1]
        return input_shape[0],  self.features_dim
    
    
from keras.layers import Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)