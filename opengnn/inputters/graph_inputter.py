from typing import Tuple, Callable, Dict, Any

import numpy as np

import tensorflow as tf

from opengnn.inputters.inputter import Inputter
from opengnn.inputters.token_embedder import TokenEmbedder
from opengnn.inputters.copying_token_embedder import CopyingTokenEmbedder
from opengnn.utils.data import diverse_batch, get_padded_shapes
from opengnn.utils.misc import find_first, count_lines


class GraphInputter(Inputter):
    def __init__(self,
                 edge_vocabulary_file_key,
                 allow_unk_edges: bool = True):
        super().__init__()
        self.vocabulary_file_key = edge_vocabulary_file_key
        self.num_unk_edges = 1 if allow_unk_edges else 0

    def initialize(self, metadata):
        super().initialize(metadata)
        self.vocabulary_file = metadata[self.vocabulary_file_key]

        self._num_edge_types = count_lines(
            self.vocabulary_file) + self.num_unk_edges
        self.edge_vocabulary = tf.contrib.lookup.index_table_from_file(
            self.vocabulary_file,
            vocab_size=self.num_edge_types - self.num_unk_edges,
            num_oov_buckets=self.num_unk_edges)

    def extract_metadata(self, data_file):
        self._node_features_size = find_first(
            data_file, lambda sample: len(sample["node_features"][0]))

    def extract_tensors(self) -> Tuple[Callable[[Dict[str, Any]], Dict[str, np.ndarray]], Dict[str, tf.DType], Dict[str, tf.TensorShape]]:
        # extract graphs and initial node_features
        def _tensor_extractor(sample):
            edge_labels, edges = [], []
            for edge in sample['edges']:
                edge_labels.append(edge[0])
                edges.append(edge[1:])
            return {
                "edges": edges,
                "edge_labels": edge_labels,
                "features": sample['node_features']
            }

        # types and shapes
        tensor_types = {
            "edges": tf.int64,
            "edge_labels": tf.string,
            "features": tf.float32
        }
        tensor_shapes = {
            "edges": tf.TensorShape([None, 2]),
            "edge_labels": tf.TensorShape([None]),
            "features": tf.TensorShape([None, self.node_features_size])
        }

        return _tensor_extractor, tensor_types, tensor_shapes

    def _process(self, data: Dict[str, Any], input_data)-> Dict[str, Any]:
        data["length"] = tf.shape(data["features"])[0]
        num_nodes = tf.cast(data["length"], tf.int64)

        edges = data["edges"]
        edge_types = self.edge_vocabulary.lookup(data['edge_labels'])
        edges = tf.concat([tf.expand_dims(edge_types, -1), edges], axis=1)

        if not bool(self.num_unk_edges):
            edges = tf.py_func(
                lambda edges: np.array([edge for edge in edges if edge[0] != -1]),
                inp=(edges,),
                Tout=(tf.int64))

        values = tf.ones((tf.shape(edges)[0],), tf.int32)
        shape = (self.num_edge_types, num_nodes, num_nodes)
        data['graph'] = tf.SparseTensor(edges, values, shape)
        del data['edges']
        del data['edge_labels']
        return data

    def batch(self, dataset, batch_size):
        def _padded_batch(dataset, batch_size):
            return dataset.padded_batch(batch_size, get_padded_shapes(dataset))

        batch_fn_map = {("features", "length"): _padded_batch}
        return diverse_batch(dataset, batch_size, batch_fn_map)

    def transform(self, inputs, mode):
        return inputs

    def get_example_size(self, example):
        return example['length']

    @property
    def node_features_size(self):
        return self._node_features_size

    @property
    def num_edge_types(self):
        return self._num_edge_types


class GraphEmbedder(GraphInputter):
    def __init__(self,
                 node_embedder: TokenEmbedder,
                 edge_vocabulary_file_key: str,
                 truncated_graph_size: int = None,
                 allow_unk_edges: bool = True):
        super().__init__(edge_vocabulary_file_key, allow_unk_edges)
        self.node_embedder = node_embedder
        self._node_features_size = node_embedder.embedding_size
        if truncated_graph_size is not None:
            assert truncated_graph_size == node_embedder.truncated_sentence_size
        self.truncated_graph_size = truncated_graph_size

    def initialize(self, metadata):
        super().initialize(metadata)
        self.node_embedder.initialize(metadata)

    def extract_metadata(self, data_file):
        pass

    def extract_tensors(self):
        # extract necessary tensors for the word embedder
        tensor_extractor, tensor_types, tensor_shapes = \
            self.node_embedder.extract_tensors()

        # add graph information
        def _tensor_extractor(sample):
            tensor = tensor_extractor(sample["node_labels"])
            edge_labels, edges = [], []
            truncated_graph_size = len(sample["node_labels"])
            if self.truncated_graph_size is not None:
                truncated_graph_size = self.truncated_graph_size

            for edge in sample['edges']:
                if edge[1] < truncated_graph_size and edge[2] < truncated_graph_size:
                    edge_labels.append(edge[0])
                    edges.append(edge[1:])
            return {
                **tensor,
                "edges": edges,
                "edge_labels": edge_labels,
            }

        # types and shapes
        tensor_types.update({
            "edges": tf.int64,
            "edge_labels": tf.string,
        })
        tensor_shapes.update({
            "edges": tf.TensorShape([None, 2]),
            "edge_labels": tf.TensorShape([None]),
        })

        return _tensor_extractor, tensor_types, tensor_shapes

    def _process(self, data, input_data):
        data = self.node_embedder.process(data)
        data['features'] = data['ids']
        del data['ids']
        return super()._process(data, input_data)

    def batch(self, dataset, batch_size):
        def _batch(dataset, batch_size):
            return dataset.batch(batch_size)

        batch_fn_map = {"graph": _batch}
        return diverse_batch(
            dataset, batch_size, batch_fn_map,
            default_batch_fn=self.node_embedder.batch)

    def transform(self, inputs, mode):
        return self.node_embedder.transform(inputs, mode)
