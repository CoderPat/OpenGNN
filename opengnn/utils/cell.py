import tensorflow as tf
from opengnn.utils.coverage_bahdanau_attention import CoverageBahdanauAttention


def bridge_state(input_state, cell_zero_state):
    # Flattened states.
    encoder_state_flat = tf.contrib.framework.nest.flatten(input_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(cell_zero_state)

    # View encoder state as a single tensor.
    encoder_state_concat = tf.concat(encoder_state_flat, 1)

    # Extract decoder state sizes.
    decoder_state_size = []
    for tensor in decoder_state_flat:
        decoder_state_size.append(tensor.get_shape().as_list()[-1])

    decoder_total_size = sum(decoder_state_size)

    # Apply linear transformation.
    bridge = tf.layers.Dense(
        decoder_total_size,
        use_bias=False,
        name='state_bridge')

    transformed = bridge(encoder_state_concat)

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)

    # Pack as the origial decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(cell_zero_state, splitted)


def build_cell(num_units,
               num_layers,
               cell_fn,
               initial_state=None,
               copy_state=True,
               batch_size=None,
               output_dropout_rate=0.,
               input_shape=None,
               attention_mechanism_fn=None,
               memory=None,
               memory_sequence_len=None,
               alignment_history=False,
               mode=tf.estimator.ModeKeys.TRAIN,
               name=None):
    """" 
    General function to create RNN cells for decoding.
    Handles multi-layer cases, LSTMs and attention wrappers
    """
    if alignment_history == True:
        print("a")
        input()
    cells = []
    for _ in range(num_layers):
        cell = cell_fn(num_units, dtype=tf.float32, name=name)

        # build internal variables if input shape provided
        if input_shape is not None:
            cell.build(input_shape)

        # apply dropout if its a tensor or we are in training
        if ((isinstance(output_dropout_rate, tf.Tensor) or
             output_dropout_rate > 0 and mode == tf.estimator.ModeKeys.TRAIN)):
            cell = tf.contrib.rnn.DropoutWrapper(
                cell,
                output_keep_prob=1 - output_dropout_rate)

        cells.append(cell)

    if num_layers > 1:
        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    else:
        cell = cells[0]

    if initial_state is not None and not copy_state:
        if batch_size is None:
            batch_size = tf.shape(tf.contrib.framework.nest.flatten(initial_state)[0])[0]
        zero_state = cell.zero_state(batch_size, tf.float32)
        initial_state = bridge_state(initial_state, zero_state)

    if attention_mechanism_fn is not None:
        attention_mechanism = attention_mechanism_fn(
            num_units,
            memory,
            memory_sequence_len)

        cell_input_fn = None
        if isinstance(attention_mechanism, CoverageBahdanauAttention):
            cell_input_fn = (
                lambda inputs, attention: tf.concat([inputs, tf.split(attention, 2, axis=-1)[0]], -1))

        cell = tf.contrib.seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            output_attention=not isinstance(
                attention_mechanism, tf.contrib.seq2seq.BahdanauAttention),
            attention_layer_size=num_units,
            initial_cell_state=initial_state,
            alignment_history=alignment_history)

        if batch_size is None:
            batch_size = tf.shape(tf.contrib.framework.nest.flatten(initial_state)[0])[0]

        initial_state = cell.zero_state(batch_size, tf.float32)

    return (cell, initial_state) if initial_state is not None else cell
