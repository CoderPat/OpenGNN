from typing import Tuple, Callable, Dict, Any
from queue import Queue

import numpy as np

import tensorflow as tf

from opengnn.inputters.inputter import Inputter
from opengnn.inputters.graph_inputter import GraphInputter, GraphEmbedder
from opengnn.utils.data import diverse_batch, get_padded_shapes
from opengnn.utils.misc import find_first, count_lines


def prune_nodes(node_labels, edges, backbone_sequence, truncated_sequence_size):
    backbone_nodes = set(backbone_sequence[:truncated_sequence_size])
    removed_nodes = set(backbone_sequence[truncated_sequence_size:])
    
    fwd_edge_list = [[] for _ in range(len(node_labels))]
    bwd_edge_list = [[] for _ in range(len(node_labels))]
    for edge in edges:
        fwd_edge_list[edge[1]].append(edge)
        bwd_edge_list[edge[2]].append(edge)

    # check nodes that still connected to a backbone_sequence
    # by doing a BFS
    # TODO: this might prove too slow
    connected = set()
    queue = Queue()
    for node in backbone_nodes:
        queue.put(node)
    while not queue.empty():
        node = queue.get()
        if node not in connected:
            connected.add(node)
            for edge in fwd_edge_list[node]:
                if edge[2] not in connected and edge[2] not in removed_nodes:
                    queue.put(edge[2])
            for edge in bwd_edge_list[node]:
                if edge[1] not in connected and edge[1] not in removed_nodes:
                    queue.put(edge[1])

    offsets, truncated_node_labels = [], []
    j = 0
    for i in range(len(node_labels)):
        offsets.append(j)
        if i in connected:
            j += 1
            truncated_node_labels.append(node_labels[i])

    truncated_backbone_sequence = []
    for i in backbone_sequence[:truncated_sequence_size]:
        truncated_backbone_sequence.append(offsets[i])

    truncated_edges = []
    for edge in edges:
        if edge[1] in connected and edge[2] in connected:
            truncated_edges.append((edge[0], offsets[edge[1]], offsets[edge[2]]))

    return truncated_node_labels, truncated_edges, truncated_backbone_sequence


class SequencedGraphInputter(Inputter):
    def __init__(self,
                 graph_inputter: GraphInputter,
                 truncated_sequence_size: int = None):
        super().__init__()
        self.graph_inputter = graph_inputter
        if truncated_sequence_size is not None and not isinstance(graph_inputter, GraphEmbedder):
            raise ValueError(
                "truncating sequences only works currently with an underlying GraphEmbedder")
        self.truncated_sequence_size = truncated_sequence_size

    def initialize(self, metadata):
        super().initialize(metadata)
        self.graph_inputter.initialize(metadata)

    def extract_metadata(self, data_file):
        pass

    def extract_tensors(self):
        # extract necessary tensors for the word embedder
        tensor_extractor, tensor_types, tensor_shapes = \
            self.graph_inputter.extract_tensors()

        # add graph information
        def _tensor_extractor(sample):
            if self.truncated_sequence_size is not None:
                sample['node_labels'], sample['edges'], sample['backbone_sequence'] = \
                    prune_nodes(
                        node_labels=sample['node_labels'],
                        edges=sample['edges'],
                        backbone_sequence=sample['backbone_sequence'],
                        truncated_sequence_size=self.truncated_sequence_size)

            tensor = tensor_extractor(sample)

            truncated_graph_size = len(sample['node_labels'])
            if self.graph_inputter.truncated_graph_size is not None:
                truncated_graph_size = self.graph_inputter.truncated_graph_size
            backbone_sequence = [idx for idx in sample['backbone_sequence']
                                 if idx < truncated_graph_size]

            return {
                **tensor,
                "primary_path": backbone_sequence}

        # types and shapes
        tensor_types.update({
            "primary_path": tf.int64,
        })
        tensor_shapes.update({
            "primary_path": tf.TensorShape([None]),
        })
        return _tensor_extractor, tensor_types, tensor_shapes

    def _process(self, data, input_data):
        """
        """
        data['primary_path_length'] = tf.shape(data['primary_path'])[0]
        return self.graph_inputter._process(data, input_data)

    def batch(self, dataset, batch_size):
        """
        """
        def _batch(dataset, batch_size):
            return dataset.padded_batch(batch_size, get_padded_shapes(dataset))

        batch_fn_map = {("primary_path", "primary_path_length"): _batch}
        return diverse_batch(dataset, batch_size, batch_fn_map,
                             default_batch_fn=self.graph_inputter.batch)

    def transform(self, inputs, mode):
        return self.graph_inputter.transform(inputs, mode)

    def get_example_size(self, example):
        return example['length']

    @property
    def node_features_size(self):
        return self.graph_inputter.node_features_size

    @property
    def num_edge_types(self):
        return self.graph_inputter.num_edge_types
