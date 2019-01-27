import tensorflow as tf
from tensorflow.contrib.seq2seq import *


def _coverage_bahdanau_score(processed_query, keys, coverage, normalize):
    """Implements Bahdanau-style (additive) scoring function.

    This attention has two forms.  The first is Bhandanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, set `normalize=True`.

    Args:
        processed_query: Tensor, shape `[batch_size, num_units]` to compare to keys.
        keys: Processed memory, shape `[batch_size, max_time, num_units]`.
        normalize: Whether to normalize the score function.

    Returns:
        A `[batch_size, max_time]` tensor of unnormalized score values.
    """
    dtype = processed_query.dtype
    # Get the number of hidden units from the trailing dimension of keys
    num_units = keys.shape[2].value or tf.shape(keys)[2]
    # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
    processed_query = tf.expand_dims(processed_query, 1)
    v = tf.get_variable(
        "attention_v", [num_units], dtype=dtype)
    w = tf.get_variable(
        "coverage_w", [num_units], dtype=dtype,)
    if normalize:
        # Scalar used in weight normalization
        g = tf.get_variable(
            "attention_g", dtype=dtype,
            initializer=tf.constant_initializer(math.sqrt((1. / num_units))),
            shape=())
        # Bias added prior to the nonlinearity
        b = tf.get_variable(
            "attention_b", [num_units], dtype=dtype,
            initializer=tf.zeros_initializer())
        # normed_v = g * v / ||v||
        normed_v = g * v * tf.rsqrt(
            tf.reduce_sum(tf.square(v)))
        return tf.reduce_sum(
            normed_v * tf.tanh(
                keys + processed_query + tf.einsum('i,jk->jki', w, coverage)), [2])
    else:
        return tf.reduce_sum(v * tf.tanh(
            keys + processed_query + tf.einsum('i,jk->jki', w, coverage)), [2])


class CoverageBahdanauAttention(BahdanauAttention):
    """Implements Bahdanau-style (additive) attention.

    This attention has two forms.  The first is Bahdanau attention,
    as described in:

    Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
    "Neural Machine Translation by Jointly Learning to Align and Translate."
    ICLR 2015. https://arxiv.org/abs/1409.0473

    The second is the normalized form.  This form is inspired by the
    weight normalization article:

    Tim Salimans, Diederik P. Kingma.
    "Weight Normalization: A Simple Reparameterization to Accelerate
     Training of Deep Neural Networks."
    https://arxiv.org/abs/1602.07868

    To enable the second form, construct the object with parameter
    `normalize=True`.
    """

    def __init__(self,
                 num_units,
                 memory,
                 memory_sequence_length=None,
                 normalize=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="CoverageBahdanauAttention"):
        """Construct the Attention mechanism.

        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.
            normalize: Python boolean.  Whether to normalize the energy term.
            probability_fn: (optional) A `callable`.  Converts the score to
                probabilities.  The default is `tf.nn.softmax`. Other options include
                `tf.contrib.seq2seq.hardmax` and `tf.contrib.sparsemax.sparsemax`.
                Its signature should be: `probabilities = probability_fn(score)`.
            score_mask_value: (optional): The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.
            dtype: The data type for the query and memory layers of the attention
                mechanism.
            name: Name to use when creating ops.
        """
        if probability_fn is None:
            probability_fn = tf.nn.softmax
        if dtype is None:
            dtype = tf.float32

        def wrapped_probability_fn(score, _): return probability_fn(score)
        super(BahdanauAttention, self).__init__(
            query_layer=tf.layers.Dense(
                num_units, name="query_layer", use_bias=False, dtype=dtype),
            memory_layer=tf.layers.Dense(
                num_units, name="memory_layer", use_bias=False, dtype=dtype),
            memory=memory,
            probability_fn=wrapped_probability_fn,
            memory_sequence_length=memory_sequence_length,
            score_mask_value=score_mask_value,
            name=name)
        self._num_units = num_units
        self._normalize = normalize
        self._name = name

    def __call__(self, query, state):
        """Score the query based on the keys and values.

        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size * 2]`
                (`alignments_size` is memory's `max_time`).

        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        coverage = tf.split(state, 2, axis=-1)[1]
        with tf.variable_scope(None, "bahdanau_attention", [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            score = _coverage_bahdanau_score(processed_query, self._keys, coverage, self._normalize)
        alignments = self._probability_fn(score, state)
        next_state = tf.concat([alignments, coverage + alignments], axis=-1)
        return alignments, next_state

    def initial_state(self, batch_size, dtype):
        """Creates the initial state values for the `AttentionWrapper` class.
        This is important for AttentionMechanisms that use the previous alignment
        to calculate the alignment at the next time step (e.g. monotonic attention).

        The default behavior is to return the same output as initial_alignments.

        Args:
            batch_size: `int32` scalar, the batch_size.
            dtype: The `dtype`.

        Returns:
            A structure of all-zero tensors with shapes as described by `state_size`.

        """
        return tf.concat([self.initial_alignments(batch_size, dtype),
                          self.initial_alignments(batch_size, dtype)],
                         axis=-1)

    @property
    def state_size(self):
        return self._alignments_size * 2
