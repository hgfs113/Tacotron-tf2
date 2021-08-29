import tensorflow as tf
from lib import *


def encoder(inputs, is_training=True, scope="encoder", reuse=None):
    """
    Args:
      inputs: A 2d tensor with shape of [N, T_x, E], with dtype of int32. Encoder inputs.
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A collection of Hidden vectors. So-called memory. Has the shape of (N, T_x, E).
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Encoder pre-net
        prenet_out = pre_net(inputs, is_training=is_training)  # (N, T_x, E/2)

        # Encoder CBHG
        enc = CBHG(prenet_out, pos='encoder', is_training)

        # Bidirectional GRU
        memory = gru(enc, num_units=128, bidirection=True) # (N, T_x, E)
    return memory



def decoder(inputs, memory, is_training=True, scope="decoder", reuse=None):
    """
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
      memory: A 3d tensor with shape of [N, T_x, E].
      is_training: Whether or not the layer is in training mode.
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns
      Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Decoder pre-net
        inputs = pre_net(inputs, is_training=is_training)  # (N, T_y/r, E/2)

        # Attention RNN
        dec, state = attention_decoder(inputs, memory, num_units=256) # (N, T_y/r, E)

        ## for attention monitoring
        alignments = tf.transpose(state.alignment_history.stack(),[1,2,0])

        # Decoder RNNs
        dec += gru(dec, 256, bidirection=False, scope="decoder_gru1") # (N, T_y/r, E)
        dec += gru(dec, 256, bidirection=False, scope="decoder_gru2") # (N, T_y/r, E)
          
        # Outputs => (N, T_y/r, n_mels*r)
        mel_hats = tf.keras.layers.dense(80*3)(dec)  # n_mels * r
    return mel_hats, alignments

def post_net(inputs, is_training=True, scope="post_net", reuse=None):
    """
    Decoder Post-processing net = CBHG
    Args:
      inputs: A 3d tensor with shape of [N, T_y/r, n_mels*r]. Log magnitude spectrogram of sound files.
        It is recovered to its original shape.
      is_training: Whether or not the layer is in training mode.  
      scope: Optional scope for `variable_scope`
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      Predicted linear spectrogram tensor with shape of [N, T_y, 1+n_fft//2].
    """
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        # Restore shape -> (N, Ty, n_mels)
        inputs = tf.reshape(inputs, [tf.shape(inputs)[0], -1, 80])

        dec = CBHG(inputs, pos='decoder', is_training)
 
        # Bidirectional GRU    
        dec = gru(dec, num_units=128, bidirection=True)  # (N, T_y, E)

        # Outputs => (N, T_y, 1+n_fft//2)
        outputs = tf.keras.layers.Dense(1+2048//2)(dec)
    return outputs
