from abc import ABC, abstractmethod

import tensorflow as tf

def get_sampling_probability(global_step,
                             read_probability=None,
                             schedule_type=None,
                             k=None):
    """Returns the sampling probability as described in
    https://arxiv.org/abs/1506.03099.

    Args:
    global_step: The training step.
    read_probability: The probability to read from the inputs.
    schedule_type: The type of schedule.
    k: The convergence constant.

    Returns:
    The probability to sample from the output ids as a 0D ``tf.Tensor`` or
    ``None`` if scheduled sampling is not configured.

    Raises:
    ValueError: if :obj:`schedule_type` is set but not :obj:`k` or if
        :obj:`schedule_type` is ``linear`` but an initial :obj:`read_probability`
        is not set.
    TypeError: if :obj:`schedule_type` is invalid.
    """
    if read_probability is None and schedule_type is None:
        return None

    if schedule_type is not None and schedule_type != "constant":
        if k is None:
            raise ValueError("scheduled_sampling_k is required when scheduled_sampling_type is set")

    step = tf.cast(global_step, tf.float32)
    k = tf.constant(k, tf.float32)

    if schedule_type == "linear":
        if read_probability is None:
            raise ValueError("Linear schedule requires an initial read probability")
        read_probability = min(read_probability, 1.0)
        read_probability = tf.maximum(read_probability - k * step, 0.0)
    elif schedule_type == "exponential":
        read_probability = tf.pow(k, step)
    elif schedule_type == "inverse_sigmoid":
        read_probability = k / (k + tf.exp(step / k))
    else:
        raise TypeError("Unknown scheduled sampling type: {}".format(schedule_type))

    return 1.0 - read_probability


class SequenceDecoder(ABC):
    """Base class for decoders."""
    @abstractmethod
    def decode(self,
               inputs: tf.Tensor,
               sequence_length: tf.Tensor,
               vocab_size: int=None,
               initial_state: tf.Tensor=None,
               sampling_probability=None,
               embedding=None,
               memory=None,
               memory_sequence_len=None,
               memory_out_ids=None,
               mode=tf.estimator.ModeKeys.TRAIN):
        """Decodes a full input sequence.

        Usually used for training and evaluation where target sequences are known.

        Args:
            inputs: The input to decode of shape :math:`[B, T, ...]`.
            sequence_length: The length of each input with shape :math:`[B]`.
            vocab_size: The output vocabulary size.
            initial_state: The initial state as a (possibly nested tuple of...) tensors.
            sampling_probability: The probability of sampling categorically from
                the output ids instead of reading directly from the inputs.
            embedding: The embedding tensor or a callable that takes word ids.
                Must be set when :obj:`sampling_probability` is set.
            memory: (optional) Memory values to query.
            memory_sequence_length: (optional) Memory values length.
            memory_out_ids: (optional) ids for original memory tokens in the output vocabylary
                Used in pointer network-like decoders
            mode: A ``tf.estimator.ModeKeys`` mode.

        Returns:
            A tuple ``(samples_ids, logits)``.
        """
        raise NotImplementedError()

    @abstractmethod
    def dynamic_decode(self,
                       embedding,
                       start_tokens,
                       end_token,
                       vocab_size=None,
                       initial_state=None,
                       output_layer=None,
                       maximum_iterations=250,
                       memory=None,
                       memory_sequence_len=None,
                       memory_out_ids=None,
                       mode=tf.estimator.ModeKeys.PREDICT):
        """Decodes dynamically from :obj:`start_tokens` with greedy search.

        Usually used for inference.

        Args:
            embedding: The embedding tensor or a callable that takes word ids.
            start_tokens: The start token ids with shape :math:`[B]`.
            end_token: The end token id.
            vocab_size: The output vocabulary size.
            initial_state: The initial state as a (possibly nested tuple of...) tensors.
            maximum_iterations: The maximum number of decoding iterations.
            mode: A ``tf.estimator.ModeKeys`` mode.
            memory: (optional) Memory values to query.
            memory_sequence_length: (optional) Memory values length.
            memory_out_ids: (optional) ids for original memory tokens in the output vocabylary
                Used in pointer network-like decoders


        Returns:
            A tuple ``(samples_ids, logits)``
        """
        raise NotImplementedError()

    @abstractmethod
    def dynamic_decode_and_search(self,
                                  embedding,
                                  start_tokens,
                                  end_token,
                                  vocab_size=None,
                                  initial_state=None,
                                  beam_width=5,
                                  length_penalty=0.0,
                                  maximum_iterations=250,
                                  memory=None,
                                  memory_sequence_len=None,
                                  memory_out_ids=None,
                                  mode=tf.estimator.ModeKeys.PREDICT):
        raise NotImplementedError()

