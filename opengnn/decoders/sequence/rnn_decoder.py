import tensorflow as tf

from opengnn.decoders.sequence.sequence_decoder import SequenceDecoder
from opengnn.utils.cell import build_cell


class RNNDecoder(SequenceDecoder):
    def __init__(self,
                 num_units,
                 num_layers,
                 copy_state=False,
                 cell_fn=tf.nn.rnn_cell.LSTMCell,
                 attention_mechanism_fn=tf.contrib.seq2seq.LuongAttention,
                 output_dropout_rate=0.,
                 coverage_loss_lambda=0.):
        self.num_units = num_units
        self.num_layers = num_layers
        self.cell_fn = cell_fn
        self.output_dropout_rate = output_dropout_rate
        self.attention_mechanism_fn = attention_mechanism_fn
        self.copy_state = copy_state
        self.coverage_loss_lambda = coverage_loss_lambda

    def decode(self,
               inputs: tf.Tensor,
               sequence_length: tf.Tensor,
               vocab_size: int = None,
               initial_state: tf.Tensor = None,
               sampling_probability=None,
               embedding=None,
               memory=None,
               memory_sequence_len=None,
               memory_out_ids=None,
               mode=tf.estimator.ModeKeys.TRAIN):

        if (sampling_probability is not None and
                (isinstance(sampling_probability, tf.Tensor) or sampling_probability > 0.0)):
            if embedding is None:
                raise ValueError("embedding argument must be set when using scheduled sampling")

            tf.summary.scalar("sampling_probability", sampling_probability)
            helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs,
                sequence_length,
                embedding,
                sampling_probability)
        else:
            helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)

        cell, initial_state = build_cell(
            self.num_units, self.num_layers,
            initial_state=initial_state,
            copy_state=self.copy_state,
            cell_fn=self.cell_fn,
            output_dropout_rate=self.output_dropout_rate,
            attention_mechanism_fn=self.attention_mechanism_fn,
            memory=memory,
            memory_sequence_len=memory_sequence_len,
            mode=mode,
            alignment_history=self.coverage_loss_lambda > 0)

        if vocab_size is not None:
            projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        else:
            projection_layer = None

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell, helper, initial_state,
            output_layer=projection_layer)

        # decode and extract logits and predictions
        outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True)
        logits = outputs.rnn_output
        ids = outputs.sample_id

        if hasattr(state, 'alignment_history') and \
           not isinstance(state.alignment_history, tuple):
            attention = tf.transpose(state.alignment_history.stack(), (1, 0, 2))
            decoder_loss = self.coverage_loss(attention, memory_sequence_len)
        else:
            decoder_loss = None

        return ids, logits, decoder_loss

    def dynamic_decode(self,
                       embedding,
                       start_tokens,
                       end_token,
                       vocab_size=None,
                       initial_state=None,
                       maximum_iterations=250,
                       memory=None,
                       memory_sequence_len=None,
                       memory_out_ids=None,
                       mode=tf.estimator.ModeKeys.PREDICT):

        cell, initial_state = build_cell(
            self.num_units, self.num_layers,
            initial_state=initial_state,
            cell_fn=self.cell_fn,
            copy_state=self.copy_state,
            output_dropout_rate=self.output_dropout_rate,
            attention_mechanism_fn=self.attention_mechanism_fn,
            memory=memory,
            memory_sequence_len=memory_sequence_len,
            mode=mode)

        if vocab_size is not None:
            projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        else:
            projection_layer = None

        # helper and decoder
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, start_tokens, end_token)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell, helper, initial_state,
            output_layer=projection_layer)

        # decode and extract logits and predictions
        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=maximum_iterations,
            swap_memory=True)
        logits = outputs.rnn_output
        ids = outputs.sample_id

        return ids, logits

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

        batch_size = tf.shape(start_tokens)[0]

        if initial_state is not None:
            initial_state = tf.contrib.seq2seq.tile_batch(
                initial_state, multiplier=beam_width)
        if memory is not None:
            memory = tf.contrib.seq2seq.tile_batch(
                memory, multiplier=beam_width)
        if memory_sequence_len is not None:
            memory_sequence_len = tf.contrib.seq2seq.tile_batch(
                memory_sequence_len, multiplier=beam_width)

        cell, initial_state = build_cell(
            self.num_units, self.num_layers,
            initial_state=initial_state,
            copy_state=self.copy_state,
            cell_fn=self.cell_fn,
            batch_size=batch_size * beam_width,
            output_dropout_rate=self.output_dropout_rate,
            attention_mechanism_fn=self.attention_mechanism_fn,
            memory=memory,
            memory_sequence_len=memory_sequence_len,
            mode=mode)

        if vocab_size is not None:
            projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        else:
            projection_layer = None

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell,
            embedding,
            start_tokens,
            end_token,
            initial_state,
            beam_width,
            output_layer=projection_layer,
            length_penalty_weight=length_penalty)

        outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)

        predicted_ids = outputs.predicted_ids[:, :, 0]
        log_probs = beam_state.log_probs[:, 0]

        return predicted_ids, log_probs

    def coverage_loss(self, attention_alignments, memory_sequence_len):
        # attention_alignments : batch x(  beam x) x t x l
        shape = tf.shape(attention_alignments)
        batch_size, output_time = shape[0], shape[2]
        cumulated_attention = tf.concat(
            [tf.zeros((batch_size, 1, output_time), dtype=tf.float32),
             tf.cumsum(attention_alignments, axis=1)[:, :-1, :]],
            axis=1)
        bounded_coverage = tf.minimum(cumulated_attention, attention_alignments)
        return self.coverage_loss_lambda * tf.reduce_sum(bounded_coverage, axis=2)
