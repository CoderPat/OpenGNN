from typing import Union

import tensorflow as tf
import opennmt as onmt

from opengnn.encoders.graph_encoder import GraphEncoder
from opengnn.utils.cell import build_cell
from opengnn.utils.ops import stack_indices, batch_gather


# initialize with weights so that it start by passing information from the RNN.
def eye_glorot(shape, dtype, partition_info):
    initial_value = tf.glorot_normal_initializer()(shape, dtype, partition_info)
    return initial_value + tf.transpose(tf.eye(shape[-1], shape[0]))


class SequencedGraphEncoder:
    def __init__(self,
                 base_graph_encoder: GraphEncoder,
                 num_units: int = None,
                 num_layers: int = None,
                 encoder_type: str = "bidirectional_rnn",
                 gnn_input_size: int = None,
                 cell_fn=tf.nn.rnn_cell.LSTMCell,
                 dropout_rate: Union[int, tf.Tensor] = 0.,
                 ignore_graph_encoder: bool = False):
        """
        Args:
            base_graph_encoder: A GraphEncoder object that represent that encoder to wrap around.
                All graph propagations will be delegated to this model.
            num_units: The number of units of the underlying path encoder. If the encoder
                is bidirectional, needs to be multiple of two
            num_layers: The number of layers of the underlying path encoder
            encoder_type: The type of underlying sequence encoder. Currently supports 'rnn' and
                'bidirectional_rnn'
            cell_fn: The type of RNN cell to use for the encoder
            bidirectional_rnn: If set, the encoder will be a composition of both two encoders
                in both directions in time
        """

        if encoder_type == "bidirectional_rnn" and num_units % 2 != 0:
            raise ValueError("num_units must be a multiple of two when using bidirectional rnns")

        self.base_graph_encoder = base_graph_encoder
        self.num_units = num_units
        self.num_layers = num_layers
        self.encoder_type = encoder_type
        self.gnn_input_size = gnn_input_size
        self.cell_fn = cell_fn
        self.dropout_rate = dropout_rate
        self.ignore_graph_encoder = ignore_graph_encoder
        self.built = False

    def build(self, node_features_size: int, num_edge_types: int,
              mode=tf.estimator.ModeKeys.TRAIN) -> None:
        if self.encoder_type == "bidirectional_rnn":
            self.fwd_cell = build_cell(
                self.num_units/2, self.num_layers,
                cell_fn=self.cell_fn,
                output_dropout_rate=self.dropout_rate,
                # input_shape=tf.TensorShape([None, node_features_size]),
                mode=mode,
                name="fwd_cell")
            self.bwd_cell = build_cell(
                self.num_units/2, self.num_layers,
                cell_fn=self.cell_fn,
                output_dropout_rate=self.dropout_rate,
                # input_shape=tf.TensorShape([None, node_features_size]),
                mode=mode,
                name="bwd_cell")
        elif self.encoder_type == "rnn":
            self.rnn_cell = build_cell(
                self.num_units, self.num_layers,
                cell_fn=self.cell_fn,
                # input_shape=tf.TensorShape([None, node_features_size]),
                output_dropout_rate=self.dropout_rate,
                mode=mode)

        self.merge_layer = tf.layers.Dense(
            self.gnn_input_size if self.gnn_input_size is not None else self.num_units,
            use_bias=False)
        self.merge_layer.build((None, None, node_features_size + self.num_units))
        self.base_graph_encoder.build(
            self.gnn_input_size if self.gnn_input_size is not None else self.num_units,
            num_edge_types)

        self.output_map = tf.layers.Dense(
            name="output_map",
            units=self.num_units,
            use_bias=False,
            kernel_initializer=eye_glorot)

        # same for state map
        # TODO: This is much easier to just do in the actual call, this will be cropped there
        # in the future needs to be put back in the build method

        self.built = True

    def __call__(self,
                 adj_matrices: tf.SparseTensor,
                 node_features: tf.Tensor,  # Shape: [ batch_size, V, D ]
                 graph_sizes: tf.Tensor,
                 primary_paths: tf.Tensor,
                 primary_path_lengths: tf.Tensor,
                 mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> tf.Tensor:

        if not self.built:
            self.build(
                node_features.shape[2].value,
                adj_matrices.get_shape()[1].value)

        # gather representations for the nodes in the pad and do decoding on this path
        primary_path_features = batch_gather(node_features, primary_paths)
        if self.encoder_type == "bidirectional_rnn":
            rnn_path_representations, rnn_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.fwd_cell,
                cell_bw=self.bwd_cell,
                inputs=primary_path_features,
                sequence_length=primary_path_lengths,
                dtype=tf.float32,
                swap_memory=True)
            rnn_path_representations = tf.concat(rnn_path_representations, axis=-1)

            # concat fwd and bwd representations in all substructures of the state
            f_rnn_state_fwd = tf.contrib.framework.nest.flatten(rnn_state[0])
            f_rnn_state_bwd = tf.contrib.framework.nest.flatten(rnn_state[1])
            f_rnn_state = [tf.concat([t1, t2], axis=-1)
                           for t1, t2 in zip(f_rnn_state_fwd, f_rnn_state_bwd)]

            rnn_state = tf.contrib.framework.nest.pack_sequence_as(rnn_state[0], f_rnn_state)

        elif self.encoder_type == "rnn":
            rnn_path_representations, rnn_state = tf.nn.dynamic_rnn(
                cell=self.rnn_cell,
                inputs=primary_path_features,
                sequence_length=primary_path_lengths,
                dtype=tf.float32,
                swap_memory=True)

        batch_size = tf.shape(node_features, out_type=tf.int64)[0]
        max_num_nodes = tf.shape(node_features, out_type=tf.int64)[1]

        # shift indices by 1 and mask padding indices to zero
        # this ensures that scatter_nd won't use a padding rnn representation over
        # the actual representation for a node with the same index as the padding value
        # by forcing scatter_nd to write padding representations into "dummy" vectors
        shifted_paths = primary_paths + 1
        shifted_paths = shifted_paths * tf.sequence_mask(primary_path_lengths, dtype=tf.int64)
        rnn_representations = tf.scatter_nd(
            indices=tf.reshape(stack_indices(shifted_paths, axis=0), (-1, 2)),
            updates=tf.reshape(rnn_path_representations, (-1, self.num_units)),
            shape=tf.stack([batch_size, max_num_nodes + 1, self.num_units], axis=0))

        # remove dummy vectors
        rnn_representations = rnn_representations[:, 1:, :]

        if self.ignore_graph_encoder:
            return rnn_representations, rnn_state

        node_representations, graph_state = self.base_graph_encoder(
            adj_matrices=adj_matrices,
            node_features=self.merge_layer(
                tf.concat([rnn_representations, node_features], axis=-1)),
            graph_sizes=graph_sizes,
            mode=mode)

        output = self.output_map(tf.concat([rnn_representations, node_representations], axis=-1))

        # flatten states (ie LSTM/multi-layer tuples) and calculate state size
        flatten_rnn_state_l = tf.contrib.framework.nest.flatten(rnn_state)
        flatten_rnn_state = tf.concat(flatten_rnn_state_l, axis=1)
        state_sizes = []
        for state in flatten_rnn_state_l:
            state_sizes.append(state.get_shape().as_list()[-1])
        total_state_size = sum(state_sizes)

        # concat graph state to this and linear map back to flatten size
        self.state_map = tf.layers.Dense(
            name="state_map",
            units=total_state_size,
            use_bias=False,
            kernel_initializer=eye_glorot)
        flatten_state = self.state_map(tf.concat([flatten_rnn_state, graph_state], axis=-1))

        # defatten
        flatten_state = tf.split(flatten_state, state_sizes, axis=1)
        state = tf.contrib.framework.nest.pack_sequence_as(rnn_state, flatten_state)
        return output, state
