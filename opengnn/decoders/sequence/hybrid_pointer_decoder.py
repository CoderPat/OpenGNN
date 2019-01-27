""" A decoder that allows pointing to input elements """

import tensorflow as tf
import numpy as np

from opengnn.decoders.sequence.rnn_decoder import RNNDecoder
from opengnn.utils.copying_wrapper import CopyingWrapper, unnormalized_luong_attention

from opengnn.utils.cell import build_cell


class HybridPointerDecoder(RNNDecoder):
    def __init__(self,
                 num_units,
                 num_layers,
                 cell_fn=tf.nn.rnn_cell.LSTMCell,
                 attention_mechanism_fn=tf.contrib.seq2seq.LuongAttention,
                 output_dropout_rate=0.,
                 copy_state=False,
                 coverage_loss_lambda=0.):
        super().__init__(
            num_units=num_units,
            num_layers=num_layers,
            cell_fn=cell_fn,
            attention_mechanism_fn=attention_mechanism_fn,
            output_dropout_rate=output_dropout_rate,
            copy_state=copy_state,
            coverage_loss_lambda=coverage_loss_lambda)

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
            vocab_size = self.num_units

        # helper and decode
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs, sequence_length, time_major=False)

        extended_vocab_size = tf.maximum(
            tf.reduce_max(memory_out_ids) + 1, tf.cast(vocab_size, tf.int64))

        copying_mechanism = unnormalized_luong_attention(
            self.num_units,
            memory,
            memory_sequence_len)

        cell = CopyingWrapper(
            cell=cell,
            copying_mechanism=copying_mechanism,
            memory_out_ids=memory_out_ids,
            extended_vocab_size=extended_vocab_size,
            output_layer=projection_layer)

        initial_state = cell.zero_state(
            tf.shape(memory)[0], tf.float32).clone(cell_state=initial_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell, helper, initial_state)

        outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            swap_memory=True)

        print(state.cell_state.alignment_history)
        if hasattr(state.cell_state, 'alignment_history') and \
                not isinstance(state.cell_state.alignment_history, tuple):
            attention = tf.transpose(state.cell_state.alignment_history.stack(), (1, 0, 2))
            decoder_loss = self.coverage_loss(attention, memory_sequence_len)
        else:
            decoder_loss = None

        logits = outputs.rnn_output
        ids = outputs.sample_id

        return ids, logits, decoder_loss

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

        cell, initial_state = build_cell(
            self.num_units, self.num_layers,
            initial_state=initial_state,
            copy_state=self.copy_state,
            cell_fn=self.cell_fn,
            output_dropout_rate=self.output_dropout_rate,
            attention_mechanism_fn=self.attention_mechanism_fn,
            memory=memory,
            memory_sequence_len=memory_sequence_len,
            mode=mode)

        if vocab_size is not None:
            projection_layer = tf.layers.Dense(vocab_size, use_bias=False)
        else:
            projection_layer = None
            vocab_size = self.num_units

        # helper and decoder
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding, start_tokens, end_token)

        extended_vocab_size = tf.maximum(
            tf.reduce_max(memory_out_ids) + 1, vocab_size)

        copying_mechanism = unnormalized_luong_attention(
            self.num_units,
            memory,
            memory_sequence_len)

        cell = CopyingWrapper(
            cell=cell,
            copying_mechanism=copying_mechanism,
            memory_out_ids=memory_out_ids,
            extended_vocab_size=extended_vocab_size,
            output_layer=projection_layer)

        initial_state = cell.zero_state(
            tf.shape(memory)[0], tf.float32).clone(cell_state=initial_state)

        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell, helper, initial_state)

        # decode and extract logits and predictions
        outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=maximum_iterations,
            swap_memory=True)
        logits = outputs.rnn_output
        predicted_ids = outputs.sample_id

        return predicted_ids, logits

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
        if memory_out_ids is not None:
            memory_out_ids = tf.contrib.seq2seq.tile_batch(
                memory_out_ids, multiplier=beam_width)

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
            vocab_size = self.num_units

        extended_vocab_size = tf.maximum(
            tf.reduce_max(memory_out_ids) + 1, vocab_size)

        copying_mechanism = unnormalized_luong_attention(
            self.num_units,
            memory,
            memory_sequence_len)

        cell = CopyingWrapper(
            cell=cell,
            copying_mechanism=copying_mechanism,
            memory_out_ids=memory_out_ids,
            extended_vocab_size=extended_vocab_size,
            output_layer=projection_layer)

        initial_state = cell.zero_state(
            tf.shape(memory)[0], tf.float32).clone(cell_state=initial_state)

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
            length_penalty_weight=length_penalty)

        outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(
            decoder, maximum_iterations=maximum_iterations)

        predicted_ids = outputs.predicted_ids[:, :, 0]
        log_probs = beam_state.log_probs[:, 0]

        return predicted_ids, log_probs
