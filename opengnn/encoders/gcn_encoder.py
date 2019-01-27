from typing import List, Union
import tensorflow as tf

from opengnn.encoders.graph_encoder import GraphEncoder


class GCNEncoder(GraphEncoder):
    """
    """

    def __init__(self,
                 layer_sizes: List[int],
                 state_dropout_rate: Union[int, tf.Tensor] = 0.,
                 use_edge_bias: bool = False):
        """
        WARNING: This class hasn't been extensively tested 

        Args:
            layer_sizes: A list with the dimensionality of each GCN layer.
            use_edge_bias: If true, messages propagated in each edge will have and edge 
              bias added
            state_dropout_rate: Dropout to be applied during training to every node represetation
              at the end of all propagations
        """
        super().__init__(len(layer_sizes), state_dropout_rate)
        self.layer_sizes = layer_sizes
        self.use_edge_bias = use_edge_bias

    def build(self, initial_node_features_size: int, num_edge_types: int,
              mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> None:
        """
        Args:
            initial_node_features_size: Dimensionality of initial node features;
              will be padded to size of node features used in GNN.
            num_edge_types: Number of edge types.
        """
        super().build(initial_node_features_size, num_edge_types)

        self._message_weights = []  # type: List[tf.Variable]
        self._edge_bias = []  # type: List[tf.Variable]

        prev_l_sizes = [initial_node_features_size] + self.layer_sizes[:-1]
        next_l_sizes = self.layer_sizes
        for layer_id, (prev_l_size, next_l_size) in enumerate(zip(prev_l_sizes, next_l_sizes)):
            # weights for the message_fun
            self._message_weights.append(tf.get_variable(
                name="message_weights_%d" % layer_id,
                shape=(self.num_edge_types, prev_l_size, next_l_size),
                initializer=tf.glorot_normal_initializer()))

            # bias for the message_fun
            if self.use_edge_bias:
                self._edge_bias.append(tf.get_variable(
                    "edge_bias_%d" % layer_id,
                    (self.num_edge_types, next_l_size)))

    def _propagate(self,
                   flattened_adj_list: tf.Tensor,
                   node_features: tf.Tensor,
                   mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
        Args:
            adj_matrices: [description]
            node_features: [description]
            graph_sizes: [description]

        Returns:
        """
        # extract information needed to calculate node degrees
        num_nodes = tf.shape(node_features)[0]
        target_nodes = flattened_adj_list[:, 2]
        data = tf.ones_like(target_nodes)

        # calculate node degrees and do propagations
        self.node_degrees = tf.unsorted_segment_sum(data, target_nodes, num_nodes)
        states = super()._propagate(flattened_adj_list, node_features, mode)

        # clean state and return
        self.node_degrees = None
        return states

    def _message_fun(self,
                     src_state,
                     tgt_state,
                     src_node_ids,
                     tgt_node_ids,
                     edge_type,
                     layer_id):
        """
        Args:
            src_state: [description]
            tgt_state: [description]
            edge_type: [description]
            layer_id: [description]

        Returns:
        """
        if self.node_degrees is None:
            raise ValueError("_message_fun was called on the wrong context")

        src_node_degrees = tf.nn.embedding_lookup(self.node_degrees, src_node_ids)
        tgt_node_degrees = tf.nn.embedding_lookup(self.node_degrees, tgt_node_ids)

        multipliers = 1 / tf.sqrt(tf.cast(src_node_degrees * tgt_node_degrees, tf.float32))

        message = tf.matmul(src_state, self._message_weights[layer_id][edge_type])
        if self.use_edge_bias:
            message += self._edge_bias[layer_id][edge_type]
        return message

    def _update_fun(self, curr_state, inc_messages, layer_id):
        for edge_type in range(self.num_edge_types):
            inc_messages = inc_messages + \
                tf.matmul(curr_state, self._message_weights[layer_id][edge_type])

        print(inc_messages.shape)
        return tf.nn.relu(inc_messages)

    def _readout_fun(self, representations):
        return representations
