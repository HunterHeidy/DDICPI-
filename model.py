from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, LSTM, Softmax, \
    Lambda, Average, Dot, Permute, Multiply, Add,Subtract,GRU
from keras.layers import Reshape, Flatten, SpatialDropout1D,  Concatenate,MaxPooling2D
from keras.models import Model
from keras.engine.topology import Layer,InputSpec

from keras.callbacks import Callback
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences

def squash(x, axis=-1):
    # s_squared_norm is really small
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x
#    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
#    scale = K.sqrt(s_squared_norm + K.epsilon())
#    return x / scale

def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = K.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)
    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))
def to_mask(x, mask, mode='mul'):
    """通用mask函数
    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
def f1(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """
    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs
class OurBidirectional(OurLayer):
    """自己封装双向RNN，允许传入mask，保证对齐
    """
    def __init__(self, layer, **args):
        super(OurBidirectional, self).__init__(**args)
        self.forward_layer = layer.__class__.from_config(layer.get_config())
        self.backward_layer = layer.__class__.from_config(layer.get_config())
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
    def reverse_sequence(self, x, mask):
        """这里的mask.shape是[batch_size, seq_len, 1]
        """
        seq_len = K.round(K.sum(mask, 1)[:, 0])
        seq_len = K.cast(seq_len, 'int32')
        return tf.reverse_sequence(x, seq_len, seq_dim=1)
    def call(self, inputs):
        x, mask = inputs
        x_forward = self.reuse(self.forward_layer, x)
        x_backward = self.reverse_sequence(x, mask)
        x_backward = self.reuse(self.backward_layer, x_backward)
        x_backward = self.reverse_sequence(x_backward, mask)
        x = K.concatenate([x_forward, x_backward], -1)
        if K.ndim(x) == 3:
            return x * mask
        else:
            return x
    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (self.forward_layer.units * 2,)
class SparseSelfAttention(OurLayer):
    """稀疏多头自注意力机制
    来自文章《Generating Long Sequences with Sparse Transformers》
    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。
    """

    def __init__(self, heads, size_per_head, rate=2,
                 key_size=None, mask_right=False, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        assert rate != 1, u'if rate=1, please use SelfAttention directly'
        self.rate = rate
        self.neighbors = rate - 1
        self.mask_right = mask_right
        self.routings = 9
        self.activation = squash

    def build(self, input_shape):
        super(SparseSelfAttention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
        # self.W = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        x = K.reshape(x, (-1, new_seq_len, seq_dim))  # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        qw = self.reuse(self.q_dense, x)
        kw = self.reuse(self.k_dense, x)
        vw = self.reuse(self.v_dense, x)
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(kw, kernel_size, self.rate)  # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, self.rate)  # shape=[None, seq_len, kernel_size, out_dim]
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 形状变换
        qw = K.reshape(qw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        kw = K.reshape(kw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        vw = K.reshape(vw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.size_per_head))
        kwp = K.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.key_size))
        vwp = K.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.size_per_head))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1, 1))
            xp_mask = K.reshape(xp_mask, (-1, new_seq_len // self.rate, self.rate, kernel_size, 1, 1))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 3, 2, 1, 4))  # shape=[None, heads, r, seq_len // r, size]
        kw = K.permute_dimensions(kw, (0, 3, 2, 1, 4))
        vw = K.permute_dimensions(vw, (0, 3, 2, 1, 4))
        qwp = K.expand_dims(qw, 4)
        kwp = K.permute_dimensions(kwp,
                                   (0, 4, 2, 1, 3, 5))  # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = K.permute_dimensions(vwp, (0, 4, 2, 1, 3, 5))
        if x_mask is not None:
            x_mask = K.permute_dimensions(x_mask, (0, 3, 2, 1, 4))
            xp_mask = K.permute_dimensions(xp_mask, (0, 4, 2, 1, 3, 5))
        # Attention1
        a = K.batch_dot(qw, kw, [4, 4]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        a = to_mask(a, x_mask, 'add')
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        if self.mask_right:
            ones = K.ones_like(a[: 1, : 1, : 1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        # Attention2
        ap = K.batch_dot(qwp, kwp, [5, 5]) / self.key_size ** 0.5
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if x_mask is not None:
            ap = to_mask(ap, xp_mask, 'add')
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if self.mask_right:
            mask = np.ones((1, kernel_size))
            mask[:, - self.neighbors:] = 0
            mask = (1 - K.constant(mask)) * 1e10
            for _ in range(4):
                mask = K.expand_dims(mask, 0)
            ap = ap - mask
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = K.concatenate([a, ap], -1)
        A = K.softmax(A)
        a, ap = A[..., : K.shape(a)[-1]], A[..., K.shape(a)[-1]:]
        # 完成输出1
        o1 = K.batch_dot(a, vw, [4, 3])
        # 完成输出2
        ap = K.expand_dims(ap, -2)
        o2 = K.batch_dot(ap, vwp, [5, 4])
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        o = to_mask(o, x_mask, 'mul')
        o = K.permute_dimensions(o, (0, 3, 2, 1, 4))
        o = K.reshape(o, (-1, new_seq_len, self.out_dim))
        o = o[:, : - pad_len]
        return self.activation(o)

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)
class Attention(OurLayer):
    """多头注意力机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):
        q, k, v = inputs[: 3]
        v_mask, q_mask = None, None
        # 这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
        if len(inputs) > 3:
            v_mask = inputs[3]
            if len(inputs) > 4:
                q_mask = inputs[4]
        # 线性变换
        qw = self.reuse(self.q_dense, q)
        kw = self.reuse(self.k_dense, k)
        vw = self.reuse(self.v_dense, v)
        # 形状变换
        qw = K.reshape(qw, (-1, K.shape(qw)[1], self.heads, self.key_size))
        kw = K.reshape(kw, (-1, K.shape(kw)[1], self.heads, self.key_size))
        vw = K.reshape(vw, (-1, K.shape(vw)[1], self.heads, self.size_per_head))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 2, 1, 3))
        kw = K.permute_dimensions(kw, (0, 2, 1, 3))
        vw = K.permute_dimensions(vw, (0, 2, 1, 3))
        # Attention
        a = K.batch_dot(qw, kw, [3, 3]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        a = to_mask(a, v_mask, 'add')
        a = K.permute_dimensions(a, (0, 3, 2, 1))
        if (self.mask_right is not False) or (self.mask_right is not None):
            if self.mask_right is True:
                ones = K.ones_like(a[: 1, : 1])
                mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
                a = a - mask
            else:
                # 这种情况下，mask_right是外部传入的0/1矩阵，shape=[q_len, k_len]
                mask = (1 - K.constant(self.mask_right)) * 1e10
                mask = K.expand_dims(K.expand_dims(mask, 0), 0)
                self.mask = mask
                a = a - mask
        a = K.softmax(a)
        self.a = a
        # 完成输出
        o = K.batch_dot(a, vw, [3, 2])
        o = K.permute_dimensions(o, (0, 2, 1, 3))
        o = K.reshape(o, (-1, K.shape(o)[1], self.out_dim))
        o = to_mask(o, q_mask, 'mul')
        return o

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.out_dim)
class SelfAttention(OurLayer):
    """多头自注意力机制
    """

    def __init__(self, heads, size_per_head, key_size=None,
                 mask_right=False, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        self.mask_right = mask_right
        # self.activation = squash
        # self.routings = 3

    def build(self, input_shape):
        super(SelfAttention, self).build(input_shape)
        self.attention = Attention(
            self.heads,
            self.size_per_head,
            self.key_size,
            self.mask_right
        )

        # self.W = self.add_weight(name='capsule_kernel',
        #                          shape=(1, input_shape[-1],
        #                                 self.heads * self.key_size),
        #                          # shape=self.kernel_size,
        #                          initializer='uniform',
        #                          trainable=True)

    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
            o = self.reuse(self.attention, [x, x, x, x_mask, x_mask])
        else:
            x = inputs
            o = self.reuse(self.attention, [x, x, x])

        # uW = K.conv1d(o, self.W)
        #
        # batch_size = K.shape(o)[0]
        # input_num_capsule = K.shape(o)[1]
        # uo = K.reshape(uW, (batch_size, input_num_capsule,
        #                     self.heads, self.key_size))
        # uo = K.permute_dimensions(uo, [0, 2, 1, 3])
        #
        # b = K.zeros_like(uo[:, :, :, 0])
        # for i in range(self.routings):
        #     b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
        #     c = K.softmax(b)
        #     c = K.permute_dimensions(c, (0, 2, 1))
        #     b = K.permute_dimensions(b, (0, 2, 1))
        #     o = K.batch_dot(c, uo, [2, 2])
        #     # o = K.permute_dimensions(o, [0, 2, 1])
        #     if i < self.routings - 1:
        #         b = b + K.batch_dot(o, uo, [2, 3])

        return o

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return (input_shape[0][0], input_shape[0][1], self.out_dim)
        else:
            return (input_shape[0], input_shape[1], self.out_dim)
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.supports_masking = True
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
                b = b + K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])
class data_generator:
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data['sent']) // self.batch_size
        # print(len(self.data['sent']))
        if len(self.data['sent']) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data['sent']))
            np.random.shuffle(list(idxs))
            sent, A, D, Y = [], [], [], []
            for i in idxs:
                dataset = self.data
                # s = dataset['sent'][i]
                # a = dataset['gene'][i]
                # d = dataset["phenotype"][i]
                # y = dataset["rel"][i]

                # s = dataset['sent'][i]
                # a = dataset['gen'][i]
                # d = dataset["disease"][i]
                # y = dataset["rel"][i]

                # s = dataset['sent'][i]
                # a = dataset['chemical'][i]
                # d = dataset["gene"][i]
                # y = dataset["rel"][i]
                # # chemical

                s = dataset['sent'][i]
                a = dataset['drug1'][i]
                d = dataset["drug2"][i]
                y = dataset["rel"][i]
                sent.append(s)
                A.append(a)
                D.append(d)
                Y.append(y)
                if len(A) == self.batch_size or i == idxs[-1]:
                    sent = seq_padding(sent)
                    A = seq_padding(A)
                    D = seq_padding(D)
                    Y = seq_padding(Y)
                    yield [sent, A, D], Y
                    [sent, A, D, Y] = [], [], [],[]
class KMaxPooling(Layer):
    """
    k-max-pooling
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.supports_masking=True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        shifted_input = tf.transpose(x, [0, 2, 1])
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        return Flatten()(top_k)
class myModel():
    def __init__(self, sent_lenth,
                 # word_embedding,
                 Lclass=2, ativ="sigmoid"):
        self.sent_lenth = sent_lenth
        # self.word_embedding = word_embedding
        self.model = None
        self.Lclass = Lclass
        self.ativ = ativ
    def attn(self,embedding):
    # def attn(self):
        print(embedding.shape)
        sent_input = Input(shape=(self.sent_lenth,))
        sent = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False,name="sent")(sent_input)
        ent1_input = Input(shape=(self.sent_lenth,))
        ent1 = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False,name="ent1")(ent1_input)
        ent2_input = Input(shape=(self.sent_lenth,))
        ent2 = Embedding(input_dim=embedding.shape[0], output_dim=embedding.shape[1], weights=[embedding], trainable=False,name="ent2")(ent2_input)

        # sent_input = Input(shape=(self.sent_lenth,))
        # sent = Embedding(input_dim=20000, output_dim=200, trainable=False, name="word")(sent_input)
        # ent1_input = Input(shape=(self.sent_lenth,))
        # ent1 = Embedding(input_dim=20000, output_dim=200, trainable=False, name="ent1")(ent1_input)
        # ent2_input = Input(shape=(self.sent_lenth,))
        # ent2 = Embedding(input_dim=20000, output_dim=200, trainable=False, name="ent2")(ent2_input)
        #

        mask_s = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(sent_input)
        mask_e1 = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(ent1_input)
        mask_e2 = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(ent2_input)


        # sent = OurBidirectional(LSTM(100, recurrent_dropout=0.25, activation='relu', return_sequences=True))([sent,mask_s])
        s1 = SparseSelfAttention(20, 10)([sent,mask_s])
        s2 = Capsule(num_capsule=self.sent_lenth, dim_capsule=200, routings=3)(sent)
        s2 = Dropout(0.25)(s2)
        e1 =  OurBidirectional(LSTM(50,return_sequences=True))([ent1, mask_e1])
        e2 =  OurBidirectional(LSTM(50,return_sequences=True))([ent2, mask_e2])# activation='relu',
        e = Concatenate(axis=2)([e1,e2])
        sent_ent1 = Add()([s1, e])
        sent_ent2 = Add()([s2, e])
        output = Concatenate()([sent_ent1,sent_ent2])
        output = KMaxPooling(k=3)(output)
        # output = Flatten()(output)
        output = Dense(units=128,#64
                       kernel_regularizer=regularizers.l2(0.002)#(L1)
                       )(output)#(RDD 不适用 ))
        output = Activation("relu")(output)
        # at2
        # sent =  OurBidirectional(LSTM(100, recurrent_dropout=0.25, activation='relu', return_sequences=True))([sent,mask_s])
        # s1 = SparseSelfAttention(20, 10)([sent,mask_s])
        # s2 = Capsule(num_capsule=140, dim_capsule=200, routings=3)(s1)
        # s3 = Add()([s1,s2])
        # s3 = Dropout(0.2)(s3)
        # output = KMaxPooling(k=3)(s3)
        # output = Dense(units=64,
        #                kernel_regularizer=regularizers.l2(0.001)  # (L1)
        #                )(output)  # (RDD 不适work.errors_impl.InvalidArgumentError: Inc用 ))
        # output = Activation("relu")(output)



        output = Dense(units=self.Lclass,kernel_regularizer=regularizers.l2(0.002))(output)
        output = Activation("softmax")(output)
        model = Model(inputs=[sent_input, ent1_input, ent2_input], outputs=[output])
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['acc',f1])
        model.summary()
        self.model = model
    def train(self, inputs, batch_size, epochs,maxSentLen):
        if self.model:
            # early_stopper = EarlyStopping(patience=50, verbose=0)
            # # check_pointer = ModelCheckpoint(save_path, verbose=0, save_best_only=True)
            # clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))#callbacks=callbacks
            # clr = CyclicLR(base_lr=0.0006, max_lr=0.001,
            #                     step_size=4000, scale_fn=clr_fn,
            #                     scale_mode='iterations')


            inputs["sent"] = pad_sequences(inputs["sent"], maxlen=maxSentLen, padding="post")
            inputs["rel"] = np.array(inputs["rel"])

            # inputs["gene"] = pad_sequences(inputs["gene"], maxlen=maxSentLen, padding="post")
            # inputs["phenotype"] = pad_sequences(inputs["phenotype"], maxlen=maxSentLen, padding="post")

            # inputs["gen"] = pad_sequences(inputs["gen"], maxlen=maxSentLen, padding="post")
            # inputs["disease"] = pad_sequences(inputs["disease"], maxlen=maxSentLen, padding="post")

            inputs["drug1"] = pad_sequences(inputs["drug1"], maxlen=maxSentLen, padding="post")
            inputs["drug2"] = pad_sequences(inputs["drug2"], maxlen=maxSentLen, padding="post")

            # inputs["gene"] = pad_sequences(inputs["gene"], maxlen=maxSentLen, padding="post")
            # inputs["chemical"] = pad_sequences(inputs["chemical"], maxlen=maxSentLen, padding="post")

            # callbacks=[early_stopper,clr]
            train_data =data_generator(inputs,batch_size)
            self.model.fit_generator(
                                train_data.__iter__(),
                                steps_per_epoch=len(train_data),
                                epochs=epochs,
                                # callbacks=callbacks

                                # validation_data=valid_D.__iter__(),
                                # validation_steps=len(valid_D)
            )

            # self.model.fit(inputs,
            #                validation_split=validation_split,
            #                batch_size=batch_size,
            #                epochs=epochs,
            #                verbose=verbose,
            #                callbacks=callbacks
            #                )
            print('\n---------- train done ----------')
        else:
            print("model error")
    def predict(self, inputs):
        if self.model:
            output = self.model.predict(inputs, verbose=1)
            print('\n---------- predict done ----------')
            return output
        else:
            print("model error")
