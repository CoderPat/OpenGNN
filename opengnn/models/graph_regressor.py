from typing import List, Dict, Tuple

import tensorflow as tf

from opengnn.inputters.features_inputter import FeaturesInputter
from opengnn.inputters.graph_inputter import GraphInputter
from opengnn.encoders.graph_encoder import GraphEncoder
from opengnn.models.model import Model


class GraphRegressor(Model):
    def __init__(self,
                 source_inputter: GraphInputter,
                 target_inputter: FeaturesInputter,
                 encoder: GraphEncoder,
                 name: str)-> None:
        super().__init__(name, source_inputter, target_inputter)
        self.encoder = encoder

    def __call__(self,
                 features: Dict[str, tf.Tensor],
                 labels,
                 mode,
                 params,
                 config=None)-> Tuple[tf.Tensor, tf.Tensor]:

        adj_matrices = features["graph"]
        node_features = features["features"]
        graph_sizes = features["length"]

        # format input features (ex: embedding labels)
        node_features = self.features_inputter.transform(
            (node_features, graph_sizes), mode)

        with tf.variable_scope("encoder"):
            # build encoder using inputter metadatza manually
            # this is due to https://github.com/tensorflow/tensorflow/issues/15624
            # and to the way estimators need to rebuild variables
            self.encoder.build(
                self.features_inputter.node_features_size,
                self.features_inputter.num_edge_types,
                mode=mode)

            node_representations, graph_state = self.encoder(
                adj_matrices=adj_matrices,
                node_features=node_features,
                graph_sizes=graph_sizes,
                mode=mode)

        output_layer = tf.layers.Dense(
            self.labels_inputter.features_size,
            name="node_output_layer")

        output = output_layer(graph_state)
        return output, output

    def regress(self,
                initial_representations,
                final_representations,
                graph_sizes):
        output_size = self.labels_inputter.features_size

        # extract info for flattening and deflattentingfinal_repr_size
        final_repr_size = final_representations.shape[-1]
        initial_repr_size = initial_representations.shape[-1]
        new_shape = tf.concat([tf.shape(initial_representations)[:2],
                               (output_size,)], axis=0)

        # obtain concatenation of final and initial representations and flatten for use in layers
        all_representation = tf.concat(
            [initial_representations, final_representations], axis=2)
        flat_all_repr = tf.reshape(
            all_representation, (-1, initial_repr_size + final_repr_size))
        flat_final_repr = tf.reshape(
            final_representations, (-1, final_repr_size))

        # build gate layer (an aproximated form of attention to determine
        # importance of nodes representations) and output layer
        gate_layer = tf.layers.Dense(
            1,
            activation=tf.nn.sigmoid,
            name="node_gate_layer")
        output_layer = tf.layers.Dense(
            output_size,
            name="node_output_layer")

        # calculate weighted, node-level outputs
        flat_node_outputs = tf.expand_dims(gate_layer(
            flat_all_repr), 1) * output_layer(flat_final_repr)
        node_outputs = tf.reshape(flat_node_outputs, new_shape)

        graph_mask = tf.expand_dims(
            tf.sequence_mask(graph_sizes, dtype=tf.float32), -1)
        return tf.reduce_sum(node_outputs * graph_mask, axis=1)

    def compute_loss(self, _, labels, outputs, params, mode: tf.estimator.ModeKeys)-> tf.Tensor:
        labels_vectors = labels['features']
        batch_size = tf.cast(tf.shape(labels_vectors)[0], tf.float32)

        diff = labels_vectors - outputs
        loss = tf.reduce_sum(0.5 * tf.square(diff)) / batch_size
        return loss, loss

    def compute_metrics(self, _, labels, predictions):
        labels_vectors = labels['features']
        eval_metric_ops = {}

        eval_metric_ops["mean_absolute_error"] = tf.metrics.mean_absolute_error(
            labels_vectors, predictions)

        return eval_metric_ops
