import tensorflow as tf
import tensorflow_addons as tfa

def char_embed(inputs, vocab_size, num_units, zero_pad=True, scope="embedding", reuse=None):
    """
    Embeds a given tensor. 
    TODO description char_embed, mb remove scope
    Returns:
      A `Tensor` with one more rank than inputs's. The last dimesionality
        should be `num_units`.
    """
    lookup_table = tf.Variable(tf.random.normal([vocab_size, num_units], mean=0.0, stddev=0.01),
            shape=[vocab_size, num_units], dtype=tf.float32)
    if zero_pad:
        lookup_table = tf.concat((tf.zeros(shape=[1, num_units]), 
                                  lookup_table[1:, :]), 0)
    return tf.nn.embedding_lookup(lookup_table, inputs)


def batch_norm(inputs, is_training=True, activation_fn=None, scope="batch_norm", reuse=None):
    """
    TODO description pre_net
    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    """
    inputs_shape = inputs.get_shape()
    inputs_rank = inputs_shape.ndims

    # use fused batch norm if inputs_rank in [2, 3, 4] as it is much faster.
    # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    if inputs_rank in [2, 3, 4]:
        if inputs_rank == 2:
            inputs = tf.expand_dims(inputs, axis=1)
            inputs = tf.expand_dims(inputs, axis=2)
        elif inputs_rank == 3:
            inputs = tf.expand_dims(inputs, axis=1)

        outputs = tf.keras.layers.BatchNormalization()(inputs)
        # outputs = tf.contrib.layers.batch_norm(inputs=inputs, center=True, scale=True, updates_collections=None,
        #                                       is_training=is_training, scope=scope, fused=True, reuse=reuse)
        # restore original shape
        if inputs_rank == 2:
            outputs = tf.squeeze(outputs, axis=[1, 2])
        elif inputs_rank == 3:
            outputs = tf.squeeze(outputs, axis=1)
    else:
        outputs = tf.keras.layers.BatchNormalization()(inputs)
        # outputs = tf.contrib.layers.batch_norm(inputs=inputs, center=True, scale=True, updates_collections=None,
        #                                       is_training=is_training, scope=scope, reuse=reuse, fused=False)
    if activation_fn is not None:
        outputs = activation_fn(outputs)
    return outputs


def pre_net(input, num_units=[256, 128], dropout=0.5, scope='pre_net', is_training=True):
    """
    TODO description pre_net, mb remove scope
    """
    output = tf.keras.layers.Dense(num_units[0], activation='relu', name="dense1")(input)
    output = tf.keras.layers.Dropout(rate=dropout, name="dropout1")(output)
    output = tf.keras.layers.Dense(num_units[1], activation='relu', name="dense2")(output)
    output = tf.keras.layers.Dropout(rate=dropout, name="dropout2")(output)
    return output


def conv1d(inputs, filters=None, size=1, rate=1, padding="same", use_bias=False,
           activation_fn=None, scope="conv1d", reuse=None):
    """
    TODO: description
    Produces convolution 1d
    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal`
      use_bias: A boolean.
    """
    with tf.compat.v1.variable_scope(scope):
        if padding == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"
        if filters is None:
            filters = inputs.get_shape().as_list[-1]
        params = {
                    "dilation_rate": rate,
                    "padding": padding,
                    "activation": activation_fn,
                    "use_bias": use_bias
                }
        outputs = tf.keras.layers.Conv1D(filters, size, **params)(inputs)
    return outputs


def conv1d_banks(inputs, K=16, is_training=True, scope="conv1d_banks", reuse=None):
    """
    Applies a series of conv1d separately.

    Args:
      inputs: A 3d tensor with shape of [N, T, C]
      K: An int. The size of conv1d banks. That is,
        The `inputs` are convolved with K filters: 1, 2, ..., K.
      is_training: A boolean. This is passed to an argument of `batch_norm`.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A 3d tensor with shape of [N, T, K*Hp.embed_size//2].
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        outputs = []
        for k in range(1, K + 1):  # k = 2...K
            with tf.compat.v1.variable_scope(f'num_%d' % k):
                outputs.append(
                        conv1d(inputs, 128, k)  # embed_size=256 // 2
                    )
        outputs = tf.concat((outputs, output), -1)
        outputs = batch_norm(outputs, is_training=is_training, activation_fn=tf.nn.relu)
    return outputs  # (N, T, embed_size//2*K)


def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    """
    Applies a GRU. TODO (description)

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results
        are concatenated.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    """
    if num_units is None:
        num_units = inputs.get_shape().as_list[-1]
    rnn = tf.keras.layers.GRU(num_units)
    if bidirection:
        outputs = tf.keras.layers.Bidirectional(layer=rnn)(inputs)
    else:
        outputs = rnn(inputs)
    return outputs


def highway_net(inputs, num_units=None, scope="highwaynet", reuse=None):
    """
    TODO description
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        if not num_units:
            num_units = inputs.get_shape()[-1]
        T = tf.keras.layers.Dense(num_units, activation=tf.nn.sigmoid, name="dense1")(inputs)
        H = tf.keras.layers.Dense(num_units, activation=tf.nn.relu, name="dense2")(inputs)
        outputs = T * H + (1.0 - T) * inputs
        return outputs


def attention_decoder(inputs, memory, num_units=None, scope="attention_decoder", reuse=None):
    """
    Applies a GRU to `inputs`, while attending `memory`.
    Args:
      inputs: A 3d tensor with shape of [N, T', C']. Decoder inputs.
      memory: A 3d tensor with shape of [N, T, C]. Outputs of encoder network.
      num_units: An int. Attention size.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    
    Returns:
      A 3d tensor with shape of [N, T, num_units].    
    """
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list[-1]

        attention_mechanism = tfa.seq2seq.BahdanauAttention(num_units, memory)
        cell_with_attention = tfa.seq2seq.AttentionWrapper(tf.keras.layers.GRUCell(num_units),
                attention_mechanism, num_units, alignment_history=True)

        output, state = tf.keras.layers.RNN(cell_with_attention, return_state=True)(inputs)
    return outputs, state

def CBHG(inputs, pos='encoder', is_training=None):
    is_decoder = pos != 'encoder'
    if is_decoder:
        K = 8
        filters = [128, 80]
    else:
        K = 16
        filters = [128, 128]

    # Conv1D banks
    outputs = conv1d_banks(inputs, K=K, is_training=is_training)  # (N, T_x, K*E/2)

    # Max pooling
    outputs = tf.keras.layers.MaxPool1D(pool_size=2, strides=1, padding="same")(outputs)  # (N, T_x, K*E/2)

    # Conv1D projections
    outputs = conv1d(outputs, filters=filters[0], size=3, scope="conv1d_1")  # (N, T_x, E/2)
    outputs = batch_norm(outputs, is_training=is_training, activation_fn=tf.nn.relu, scope="conv1d_1")

    outputs = conv1d(outputs, filters=filters[1], size=3, scope="conv1d_2")  # (N, T_x, E/2)
    outputs = batch_norm(outputs, is_training=is_training, scope="conv1d_2")

    if is_decoder:
        # Extra affine transformation for dimensionality sync
        outputs = tf.keras.layers.Dense(128) (outputs)
    else:
        # Residual connections
        outputs += inputs # (N, T_x, E/2)

    # Highway Nets
    for i in range(4):
        outputs = highway_net(outputs, num_units=128, scope=f'highwaynet_%d' % i)  # (N, T_x, E/2)
    return outputs
