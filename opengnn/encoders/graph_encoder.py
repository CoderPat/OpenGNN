from abc import ABC, abstractmethod
from typing import Union

import tensorflow as tf


class GraphEncoder(ABC):
    def __init__(self,
                 num_layers: int,
                 state_dropout_rate: Union[int, tf.Tensor] = 0.,
                 create_bwd_edges: bool = True,
                 tie_bwd_edges: bool = False,
                 gated_state: bool = True):
        """
        Args:
            num_layers: The number of layers or timesteps to do propagations. This notion of layer
              is relative to the specific encoder.
            state_dropout_rate: A int or tensor indicating the probability of droping units in
              the node state at then end of apropagation. If its an int, this dropout will only be 
              applied when ```mode = tf.estimator.ModeKeys.TRAIN```
            create_bwd_edges: Boolean specifying if the encoder should add artificial
              edges corresponding to the original edges reversed. This potentially helps to
              propagate information more extensively throughout the graph at the cost of more
              time per propagation step.
            tie_bwd_edges: Boolean specifying if the artificial reversed edges should share the
              weights with the original foward edges. This can potentially save the model 
              memory/computational time (depending on the specify encoder) at the cost of 
              potentially reducing expressivess when compared to untied weights.
        """
        self.num_layers = num_layers
        self.state_dropout_rate = state_dropout_rate
        self.create_bwd_edges = create_bwd_edges
        self.tie_bwd_edges = tie_bwd_edges
        self.gated_state = gated_state
        self.built = False

    def build(self, node_features_size: int, num_edge_types: int,
              mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> None:
        """
        Builds the Encoder, i.e., prepares variables and layers used when the
        encoder gets called. Subclasses should override this to prepare needed data.

        Args:
            node_features_size: Dimensionality of initial node features;
              will be padded to layer_size.
            num_edge_types: Number of edge types.
        """
        self.node_features_size = node_features_size
        self.num_edge_types = num_edge_types
        if self.create_bwd_edges and not self.tie_bwd_edges:
            self.num_edge_types *= 2

        self.built = True

    def __call__(self,
                 adj_matrices: tf.SparseTensor,
                 node_features: tf.Tensor,  # Shape: [ batch_size, V, D ]
                 graph_sizes: tf.Tensor,
                 mode: tf.estimator.ModeKeys = tf.estimator.ModeKeys.TRAIN) -> tf.Tensor:
        """
        Encode graphs given by a (sparse) adjacency matrix and and their initial node features,
        returning the encoding of all graph nodes.

        Args:
            adj_matrices: SparseTensor of dense shape
              [BatchSize, NumEdgeTypes, MaxNumNodes, MaxNumNodes] representing edges in graph.
              adj_matrices[g, e, v, u] == 1 means that in graph g, there is an edge of
              type e between v and u.
            node_features: Tensor of shape [BatchSize, MaxNumNodes, NodeFeatureDimension],
              representing initial node features. node_features[g, v, :] are the features of
              node v in graph g.
            graph_sizes: Tensor of shape [BatchSize] representing the number of used nodes in
              the batchedand padded graphs. graph_size[g] is the number of nodes in graph g.
            mode: Flag indicating run mode. [Unused]

        Returns: 
            Tensor of shape [BatchSize, MaxNumNodes, NodeFeatureDimension]. Representations for 
              padding nodes will be zero vectors
        """
        if not self.built:
            self.build(
                node_features_size=node_features.shape[2].value,
                num_edge_types=adj_matrices.get_shape()[1].value)

        if self.create_bwd_edges:
            adj_matrices = self._create_backward_edges(adj_matrices)

        # We only care about the edge indices, as adj_matrices is only an indicator
        # matrix with values 1 or not-present (i.e., an adjacency list):
        # Shape: [ num of edges (not edge types) ~ E, 4 ]
        adj_list = tf.cast(adj_matrices.indices, tf.int32)

        max_num_vertices = tf.shape(node_features, out_type=tf.int32)[1]
        total_edges = tf.shape(adj_list, out_type=tf.int32)[0]

        # Calculate offsets for flattening the adj matrices, as we are merging all graphs into one big graph.
        # Nodes in first graph are range(0,MaxNumNodes) and edges are shifted by [0,0],
        # nodes in second graph are range(MaxNumNodes,2*MaxNumNodes) and edges are
        # shifted by [MaxNumNodes,MaxNumNodes], etc.
        graph_ids_per_edge = adj_list[:, 0]
        node_id_offsets_per_edge = tf.expand_dims(graph_ids_per_edge, axis=-1) * max_num_vertices
        edge_shifts_per_edge = tf.tile(node_id_offsets_per_edge, multiples=(1, 2))
        offsets_per_edge = tf.concat(
            [tf.zeros(shape=(total_edges, 1), dtype=tf.int32),  # we don't need to shift the edge type
             edge_shifts_per_edge],
            axis=1)

        # Flatten both adj matrices and node features. For the adjacency list, we strip out the graph id
        # and instead shift the node IDs in edges.
        flattened_adj_list = offsets_per_edge + adj_list[:, 1:]
        flattened_node_features = tf.reshape(
            node_features,
            shape=(-1, self.node_features_size))

        # propagate on this big graph and unflatten representations
        flattened_node_repr = self._propagate(flattened_adj_list, flattened_node_features, mode)
        node_representations = tf.reshape(
            flattened_node_repr,
            shape=(-1, max_num_vertices, flattened_node_repr.shape[-1]))

        # mask for padding nodes
        graph_mask = tf.expand_dims(tf.sequence_mask(graph_sizes, dtype=tf.float32), -1)
        if self.gated_state:
            gate_layer = tf.layers.Dense(
                1,
                activation=tf.nn.sigmoid,
                name="node_gate_layer")

            output_layer = tf.layers.Dense(
                node_representations.shape[-1],
                name="node_output_layer")

            # calculate weighted, node-level outputs
            node_all_repr = tf.concat([node_features, node_representations], axis=-1)
            graph_state = gate_layer(node_all_repr) * output_layer(node_representations)
            graph_state = tf.reduce_sum(graph_state * graph_mask, axis=1)

        else:
            graph_state = tf.reduce_sum(node_representations * graph_mask, axis=1)
            graph_state /= tf.cast(tf.expand_dims(graph_sizes, 1), tf.float32)

        return node_representations, graph_state

    def _propagate(self,
                   flattened_adj_list: tf.Tensor,
                   node_features: tf.Tensor,
                   mode: tf.estimator.ModeKeys) -> tf.Tensor:
        """
        Args:
            flattened_adj_list: Tensor of shape [NumEdgesInBatch, 3] representing all edges in batch.
              flattened_adj_list[e, v, u] means that there is an edge of type between v and u.
              Node IDs range between 0 and MaxNumNodes*BatchSize.
            node_features: Tensor of shape [MaxNumNodes*BatchSize, NodeFeatureDimension],
              representing initial node features. node_features[v, :] are the initial features of node v.

        Returns:
            Tensor of shape [MaxNumNodes*BatchSize, NodeFeatureDimension],
        """
        edges_per_type = []  # type: List[tf.Tensor]

        # separate adjacency matrices per edge type to save later computations
        # should this be done by building a sparse matrix and slicing?
        for edge_id in range(self.num_edge_types):
            edges_ind = tf.where(tf.equal(flattened_adj_list[:, 0], edge_id))[:, -1]
            edges_per_type.append(tf.nn.embedding_lookup(
                params=flattened_adj_list[:, 1:], ids=edges_ind))

        num_vertices = tf.shape(node_features, out_type=tf.int32)[0]
        node_states = node_features
        for layer_idx in range(self.num_layers):
            with tf.variable_scope("gnn_layer_%i" % layer_idx):
                messages_per_edge = []
                for edge_type_id in range(self.num_edge_types):

                    # collect hidden states of source and target nodes
                    source_node_ids = edges_per_type[edge_type_id][:, 0]
                    target_node_ids = edges_per_type[edge_type_id][:, 1]

                    source_states = tf.nn.embedding_lookup(
                        params=node_states, ids=source_node_ids)

                    target_states = tf.nn.embedding_lookup(
                        params=node_states, ids=target_node_ids)

                    # calculate all messages
                    messages = self._message_fun(
                        source_states,
                        target_states,
                        source_node_ids,
                        target_node_ids,
                        edge_type_id,
                        layer_idx)

                    # aggregate messages per target nodes
                    # TODO: Should the aggregation method also be overridable
                    # (e.g., by max pooling, attention, ...)?
                    messages_per_edge.append(tf.unsorted_segment_sum(
                        data=messages,
                        segment_ids=edges_per_type[edge_type_id][:, 1],
                        num_segments=num_vertices))

                incoming_messages = tf.add_n(messages_per_edge)
                # update states
                node_states = self._update_fun(node_states, incoming_messages, layer_idx)

                if (isinstance(self.state_dropout_rate, tf.Tensor) or
                        self.state_dropout_rate > 0 and mode == tf.estimator.ModeKeys.TRAIN):
                    node_states = tf.layers.dropout(
                        node_states, rate=self.state_dropout_rate)

        return self._readout_fun(node_states)

    def _create_backward_edges(self, adj_matrices):
        fwd_edges = adj_matrices.indices
        values = adj_matrices.values
        shape = adj_matrices.dense_shape
        reverse_edges = [fwd_edges[:, 3], fwd_edges[:, 2]]
        if self.tie_bwd_edges:
            bwd_edges = tf.stack(
                [fwd_edges[:, 0], fwd_edges[:, 1]] + reverse_edges,
                axis=1)
        else:
            bwd_edges = tf.stack(
                [fwd_edges[:, 0], fwd_edges[:, 1] + self.num_edge_types//2] + reverse_edges,
                axis=1)
            shape = tf.stack(
                [shape[0], self.num_edge_types, shape[2], shape[3]],
                axis=0)

        # NOTE: This only works since currently there is no use for the values of this tensor.
        # If later we introduce weighted edges this might need to change.
        fwd_graph = tf.SparseTensor(fwd_edges, values, shape)
        bwd_graph = tf.SparseTensor(bwd_edges, values, shape)
        return tf.sparse_add(fwd_graph, bwd_graph)

    @abstractmethod
    def _message_fun(self,
                     source_states: tf.Tensor,
                     target_states: tf.Tensor,
                     source_node_ids: tf.Tensor,
                     target_node_ids: tf.Tensor,
                     edge_type_id: int,
                     layer_idx: int) -> tf.Tensor:
        """
        Args:
            source_states: Tensor of shape [NumEdgesOfTypeInBatch, NodeFeatureDimension] 
              representing current node representation of source nodes of edges.
            target_states: Tensor of shape [NumEdgesOfTypeInBatch, NodeFeatureDimension] 
              representing current node representation of target nodes of edges.
            source_node_ids: Tensor of shape [NumEdgesOfTypeInBatch] representing the ids
              of source nodes of edges
            target_node_ids: Tensor of shape [NumEdgesOfTypeInBatch] representing the ids
              of target nodes of edges
            edge_type_id: Integer id of edge type of considered edges.
            layer_idx: Current layer (helpful if different weights are used for different layers).

        Returns: Tensor of shape [NumEdgesOfTypeInBatch, MessageDimension].
        """
        raise NotImplementedError()

    @abstractmethod
    def _update_fun(self,
                    curr_states: int,
                    inc_messages: tf.Tensor,
                    layer_idx: int):
        """
        Args:
            curr_states: Tensor of shape [MaxNumNodes*BatchSize, NodeFeatureDimension]
              representing current state of nodes in graphs.
            inc_messages: Tensor of shape [MaxNumNodes*BatchSize, MessageDimension] representing
              all incoming messages, aggregated per node.
            layer_idx: Current layer (helpful if different weights are used for different layers).

        Returns: Tensor of shape [MaxNumNodes*BatchSize, NodeFeatureDimension].
        """
        raise NotImplementedError()

    @abstractmethod
    def _readout_fun(self, node_representations: tf.Tensor) -> tf.Tensor:
        """
        Args:
            curr_states: Tensor of shape [MaxNumNodes*BatchSize, NodeFeatureDimension]
              representing current state of nodes in graphs.

        Returns: Tensor of shape [MaxNumNodes*BatchSize, NodeFeatureDimension].
        """
        raise NotImplementedError()
