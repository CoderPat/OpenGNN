from typing import List, Dict, Tuple

import tensorflow as tf

import opengnn.constants as constants
from opengnn.inputters.token_embedder import TokenEmbedder
from opengnn.inputters.copying_token_embedder import CopyingTokenEmbedder
from opengnn.inputters.graph_inputter import GraphInputter
from opengnn.inputters.graph_inputter import GraphEmbedder
from opengnn.encoders.graph_encoder import GraphEncoder
from opengnn.models.model import Model
from opengnn.utils.metrics import bleu_score, rouge_2_fscore, f1_score
from opengnn.utils.misc import find
from opengnn.utils.ops import batch_gather
from opengnn.decoders.sequence.sequence_decoder import SequenceDecoder, get_sampling_probability
from opengnn.decoders.sequence.hybrid_pointer_decoder import HybridPointerDecoder


def shift_target_sequence(inputter, data: Dict[str, tf.Tensor])-> Dict[str, tf.Tensor]:
    """Prepares shifted target sequences.
    Given a target sequence ``a b c``, the decoder input should be
    ``<s> a b c`` and the output should be ``a b c </s>`` for the dynamic
    decoding to start on ``<s>`` and stop on ``</s>``.
    Args:
    inputter: The :class:`opennmt.inputters.inputter.Inputter` that processed
        :obj:`data`.
    data: A dict of ``tf.Tensor`` containing ``ids`` and ``length`` keys.
    Returns:
    The updated :obj:`data` dictionary with ``ids`` the sequence prefixed
    with the start token id and ``ids_out`` the sequence suffixed with
    the end token id. Additionally, the ``length`` is increased by 1
    to reflect the added token on both sequences.
    """
    bos = tf.cast(tf.constant([constants.START_OF_SENTENCE_ID]), tf.int64)
    eos = tf.cast(tf.constant([constants.END_OF_SENTENCE_ID]), tf.int64)

    ids = data["ids"]

    data["ids_out"] = tf.concat([ids, eos], axis=0)
    data["ids"] = tf.concat([bos, ids], axis=0)

    # Increment length accordingly.
    data["length"] += 1

    return data


def check_valid_copying_inputters(source_inputter, target_inputter):
    if (not isinstance(source_inputter, GraphEmbedder) or
            not isinstance(source_inputter.node_embedder, CopyingTokenEmbedder)):
        raise ValueError("HybridPointerDecoder requires a GraphEmbedder source inputter"
                         "with a underlying CopyingTokenEmbedder")
    if not isinstance(target_inputter, CopyingTokenEmbedder):
        raise ValueError("HybridPointerDecoder requires a CopyingTokenEmbedder target inputter")


class GraphToSequence(Model):
    def __init__(self,
                 source_inputter: GraphInputter,
                 target_inputter: TokenEmbedder,
                 encoder: GraphEncoder,
                 decoder: SequenceDecoder,
                 name: str,
                 metrics: Tuple[str] = ('BLEU', 'ROUGE')):
        super().__init__(name, source_inputter, target_inputter)
        self.encoder = encoder
        self.decoder = decoder
        self.metrics = metrics
        self.labels_inputter.add_process_hooks([shift_target_sequence])
        self.use_copying = isinstance(decoder, HybridPointerDecoder)
        if self.use_copying:
            check_valid_copying_inputters(source_inputter, target_inputter)

    def __call__(self,
                 features: Dict[str, tf.Tensor],
                 labels,
                 mode,
                 params,
                 config=None)-> Tuple[tf.Tensor, tf.Tensor]:
        """
        Args:
            features: a dictionary containing the inputs necessary to run a graph neural network:
                a "graph" tensor represeting sparse adjacency lists, an "features" tensor 
                representing the initial features representation and a "length" tensor with
                 the size of the graphs. Optionally, if using copying, the tensors "out_ids" 
                 (representing the ids in the output vocabulary) and "pointer map" (with the map
                 for ids that point to input words)
            labels: a dictionary containing
            mode: [description]
            params: [description]
            config: [description]
        Returns:
            outputs: raw outputs
            predictions: predictions
        """
        adj_matrices = features["graph"]
        node_features = features["features"]
        graph_sizes = features["length"]
        if self.use_copying:
            # copying-specific data
            node_out_ids = features["out_ids"]
            pointer_maps = features["pointer_map"]
        else:
            node_out_ids = None
            pointer_maps = None

        target_vocab_size = self.labels_inputter.vocabulary_size

        # format input features (ex: embedding labels)
        node_features = self.features_inputter.transform(
            (node_features, graph_sizes), mode)

        # build encoder using inputter metadata manually
        # this is due to https://github.com/tensorflow/tensorflow/issues/15624
        # and to the way estimators need to rebuild variables
        self.encoder.build(
            self.features_inputter.node_features_size,
            self.features_inputter.num_edge_types,
            mode=mode)

        node_representations, initial_state = self.encoder(
            adj_matrices=adj_matrices,
            node_features=node_features,
            graph_sizes=graph_sizes,
            mode=mode)

        # function for embeddings (needs scope to properly use embeddings variables)
        def _target_embedding_fn(ids, scope):
            try:
                with tf.variable_scope(scope):
                    return self.labels_inputter.transform((ids, None), mode)
            except ValueError:
                with tf.variable_scope(scope, reuse=True):
                    return self.labels_inputter.transform((ids, None), mode)

        logits, predictions, decoder_loss = None, None, None
        # If we have labels, we can calculate the logits using teacher forcing (for loss, etc...)
        if labels is not None:
            decoder_input = labels["ids"]
            output_len = labels["length"]

            with tf.variable_scope("decoder") as scope:
                decoder_emb_input = self.labels_inputter.transform((decoder_input, None), mode)

                sampling_probability = get_sampling_probability(
                    tf.train.get_or_create_global_step(),
                    read_probability=params.get("scheduled_sampling_read_probability"),
                    schedule_type=params.get("scheduled_sampling_type"),
                    k=params.get("scheduled_sampling_k"))

                _, logits, decoder_loss = self.decoder.decode(
                    inputs=decoder_emb_input,
                    sequence_length=output_len,
                    vocab_size=target_vocab_size,
                    initial_state=initial_state,
                    sampling_probability=sampling_probability,
                    embedding=lambda ids: _target_embedding_fn(ids, scope),
                    memory=node_representations,
                    memory_sequence_len=graph_sizes,
                    memory_out_ids=node_out_ids,
                    mode=mode)

        # If we are not training, we might need to decode
        # using model predictions (for metrics, etc...)
        if mode != tf.estimator.ModeKeys.TRAIN:
            batch_size = tf.shape(node_representations)[0]
            maximum_iterations = params.get("maximum_iterations", 250)

            # If we also calculate logits using teacher forcing,
            # we need to reuse variables for this decoder
            with tf.variable_scope("decoder", reuse=labels is not None) as scope:
                # generate start/end symbol for greedy decoder
                start_tokens = tf.fill(
                    [batch_size], constants.START_OF_SENTENCE_ID)
                end_token = constants.END_OF_SENTENCE_ID

                if params.get("beam_width", 1) <= 1:
                    ids, log_probs = self.decoder.dynamic_decode(
                        embedding=lambda ids: _target_embedding_fn(ids, scope),
                        start_tokens=start_tokens,
                        end_token=end_token,
                        vocab_size=target_vocab_size,
                        maximum_iterations=maximum_iterations,
                        initial_state=initial_state,
                        memory=node_representations,
                        memory_sequence_len=graph_sizes,
                        memory_out_ids=node_out_ids,
                        mode=mode)
                else:
                    ids, log_probs = self.decoder.dynamic_decode_and_search(
                        embedding=lambda ids: _target_embedding_fn(ids, scope),
                        start_tokens=start_tokens,
                        end_token=end_token,
                        vocab_size=target_vocab_size,
                        maximum_iterations=maximum_iterations,
                        initial_state=initial_state,
                        beam_width=params.get("beam_width"),
                        length_penalty=params.get("length_penalty", 0.0),
                        memory=node_representations,
                        memory_sequence_len=graph_sizes,
                        memory_out_ids=node_out_ids,
                        mode=mode)

            # Fetch tokens from normal vocab
            rev_vocab = tf.contrib.lookup.index_to_string_table_from_file(
                self.labels_inputter.vocabulary_file,
                vocab_size=self.labels_inputter.vocabulary_size - 1,
                default_value=constants.UNKNOWN_TOKEN)

            target_tokens = rev_vocab.lookup(tf.cast(ids, tf.int64))

            if self.use_copying:
                # Fetch tokens from input sequence (non-pointer ids will get mapped to padding)
                pointer_ids = ids - target_vocab_size + 1
                pointer_ids = tf.clip_by_value(pointer_ids, 0, tf.reduce_max(pointer_ids))
                batch_size = tf.shape(pointer_maps)[0]
                padded_pointer_maps = tf.concat(
                    [tf.fill((batch_size, 1), constants.PADDING_TOKEN), pointer_maps],
                    axis=-1)
                pointer_tokens = batch_gather(padded_pointer_maps, pointer_ids)

                # Pick token from either normal vocab or input sequences depending on ids
                target_tokens = tf.where(
                    tf.greater_equal(ids, self.labels_inputter.vocabulary_size),
                    pointer_tokens,
                    target_tokens)

            predictions = {
                "tokens": target_tokens,
                "ids": ids,
                "log_probs": log_probs
            }

        return (logits, decoder_loss), predictions

    def compute_loss(self, _, labels, outputs, params, mode: tf.estimator.ModeKeys)-> tf.Tensor:
        # extract labels and batch info
        label_ids = labels["ids_out"]
        sequence_lens = labels["length"]

        outputs, decoder_loss = outputs

        batch_size = tf.shape(outputs)[0]
        batch_max_len = tf.shape(outputs)[1]

        loss_per_time = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_ids,
            logits=outputs)

        if decoder_loss is not None:
            loss_per_time = decoder_loss + loss_per_time

        weights = tf.sequence_mask(
            sequence_lens, maxlen=batch_max_len, dtype=tf.float32)

        unnorm_loss = tf.reduce_sum(loss_per_time * weights)
        total_timesteps = tf.reduce_sum(tf.cast(sequence_lens, tf.float32))

        if params.get("average_loss_in_time", False):
            loss = unnorm_loss / total_timesteps
            tb_loss = loss
        else:
            loss = unnorm_loss / tf.cast(batch_size, tf.float32)
            tb_loss = unnorm_loss / total_timesteps

        return loss, tb_loss

    def compute_metrics(self, _, labels, predictions):
        # extract labels and batch info
        labels_ids = labels["ids_out"]
        predictions_ids = predictions['ids']

        eval_metric_ops = {}

        if "BLEU" in self.metrics:
            eval_metric_ops["bleu"] = bleu_score(
                labels_ids, predictions_ids,
                constants.END_OF_SENTENCE_ID)

        if "ROUGE" in self.metrics:
            eval_metric_ops["rouge"] = rouge_2_fscore(
                labels_ids, predictions_ids,
                constants.END_OF_SENTENCE_ID)

        if "F1" in self.metrics:
            eval_metric_ops["f1"] = f1_score(
                labels_ids, predictions_ids,
                constants.END_OF_SENTENCE_ID)

        return eval_metric_ops

    def process_prediction(self, prediction):
        prediction_tokens = [token.decode('utf-8') for token in prediction['tokens']]
        cropped_tokens = prediction_tokens[:find(
            prediction_tokens, constants.END_OF_SENTENCE_TOKEN)]
        return cropped_tokens
