from typing import Union, List

import tensorflow as tf

from opengnn.encoders.graph_encoder import GraphEncoder
from opengnn.utils.cell import build_cell


class GGNNEncoder(GraphEncoder):
    def __init__(self,
                 num_timesteps: Union[int, List[int]],
                 node_feature_size: int,
                 use_edge_bias: bool = True,
                 state_dropout_rate: Union[int, tf.Tensor] = 0.,
                 gru_dropout_rate: Union[int, tf.Tensor] = 0.)-> None:
        """

        Args:
            num_timesteps: The number of timesteps in the GGNN or a list with the number of
              timesteps per GGNN layer.
            node_feature_size : The dimensionality of the node representations
            use_edge_bias: If true, messages propagated in each edge will have and edge 
              bias added
            state_dropout_rate: Dropout to be applied during training to every node represetation
              at the end of all propagations
            gru_dropout_rate: Recurrent ropout to be applied during training at every 
              node representation update
        """

        if isinstance(num_timesteps, list):
            super().__init__(sum(num_timesteps), state_dropout_rate)
            self.timesteps_per_layer = num_timesteps
        else:
            super().__init__(num_timesteps, state_dropout_rate)
            self.timesteps_per_layer = [num_timesteps]

        self.use_edge_bias = use_edge_bias
        self.node_features_size = node_feature_size
        self.gru_dropout_rate = gru_dropout_rate

    def build(self, initial_node_features_size: int, num_edge_types: int,
              mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> None:
        """
        Args:
            initial_node_features_size: Dimensionality of initial node features;
              will be padded to size of node features used in GNN.
            num_edge_types: Number of edge types.
        """
        super().build(self.node_features_size, num_edge_types)
        self.initial_node_features_size = initial_node_features_size

        if self.initial_node_features_size > self.node_features_size:
            raise ValueError("GGNN currently only support initial features size smaller "
                             "than state size")

        self._message_weights = []  # type: List[tf.Variable]
        self._update_grus = []  # type: List[tf.nn.rnn_cell.GRUCell]
        self._edge_bias = []  # type: List[tf.Variable]
        for layer_id, _ in enumerate(self.timesteps_per_layer):
            # weights for the message_fun
            self._message_weights.append(tf.get_variable(
                name="message_weights_%d" % layer_id,
                shape=(self.num_edge_types, self.node_features_size, self.node_features_size),
                initializer=tf.glorot_normal_initializer()))

            # bias for the message_fun
            if self.use_edge_bias:
                self._edge_bias.append(tf.get_variable(
                    "edge_bias_%d" % layer_id,
                    (self.num_edge_types, self.node_features_size)))

            # gru for the update_fun
            cell = build_cell(
                num_units=self.node_features_size,
                num_layers=1,
                cell_fn=tf.nn.rnn_cell.GRUCell,
                output_dropout_rate=self.gru_dropout_rate,
                input_shape=tf.TensorShape([None, self.node_features_size]),
                name="update_gru_%d" % layer_id,
                mode=mode)
            self._update_grus.append(cell)

        # final linear layer
        self.final_layer = tf.layers.Dense(self.node_features_size, use_bias=False)
        self.final_layer.build((None, self.node_features_size))

    def __call__(self,
                 adj_matrices: tf.SparseTensor,
                 node_features: tf.Tensor,
                 graph_sizes: tf.Tensor,
                 mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> tf.Tensor:
        if not self.built:
            self.build(
                node_features.shape[2].value,
                adj_matrices.get_shape()[1].value)

        # Pad features if needed
        if self.initial_node_features_size < self.node_features_size:
            pad_size = self.node_features_size - self.initial_node_features_size
            padding = tf.zeros(tf.concat(
                [tf.shape(node_features)[:2], (pad_size,)], axis=0),
                dtype=tf.float32)
            node_features = tf.concat((node_features, padding), axis=2)

        return super().__call__(adj_matrices, node_features, graph_sizes, mode)

    def _find_layer(self, unrolled_layer_id: int)-> int:
        acum = 0
        for i, timesteps in enumerate(self.timesteps_per_layer):
            acum += timesteps
            if unrolled_layer_id < acum:
                return i

    def _message_fun(self,
                     src_state,
                     tgt_state,
                     src_node_ids,
                     tgt_node_ids,
                     edge_type,
                     unrolled_layer_id):

        layer_id = self._find_layer(unrolled_layer_id)
        message = tf.matmul(src_state, self._message_weights[layer_id][edge_type])
        if self.use_edge_bias:
            message += self._edge_bias[layer_id][edge_type]
        return message

    def _update_fun(self, curr_state, inc_messages, unrolled_layer_id):
        layer_id = self._find_layer(unrolled_layer_id)
        return self._update_grus[layer_id](inc_messages, curr_state)[1]

    def _readout_fun(self, representations):
        return self.final_layer(representations)
