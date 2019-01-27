from typing import Dict, Union, Callable

import tensorflow as tf

from opengnn.inputters.inputter import Inputter
from opengnn.utils.data import shifted_batch, get_padded_shapes, diverse_batch
from opengnn.utils.misc import count_lines


class TokenEmbedder(Inputter):
    def __init__(self,
                 vocabulary_file_key,
                 embedding_size: int,
                 dropout_rate: Union[int, tf.Tensor] = 0.0,
                 truncated_sentence_size: int = None,
                 lowercase=True,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32):
        """
        Args:
            vocabulary_file ([type]): [description]
            embedding_size ([type]): [description]
            subtokens (bool, optional): Defaults to False. [description]
            trainable (bool, optional): Defaults to True. [description]
            dtype ([type], optional): Defaults to tf.float32. [description]
        """
        super().__init__()
        self.vocabulary_file_key = vocabulary_file_key
        self.embedding_size = embedding_size
        self.dropout_rate = dropout_rate
        self.truncated_sentence_size = truncated_sentence_size
        self.trainable = trainable
        self.dtype = dtype
        self.lowercase = lowercase

    def extract_tensors(self):
        def _tensor_extractor(sample):
            size = len(sample) if self.truncated_sentence_size is None else self.truncated_sentence_size
            return {"labels": [token.lower() if self.lowercase else token for token in sample][:size]}

        tensor_types = {"labels": tf.string}
        tensor_shapes = {"labels": tf.TensorShape([None])}
        return _tensor_extractor, tensor_types, tensor_shapes

    def initialize(self, metadata):
        super().initialize(metadata)
        self.vocabulary_file = metadata[self.vocabulary_file_key]

        self.vocabulary_size = count_lines(
            self.vocabulary_file) + 1
        self.vocabulary = tf.contrib.lookup.index_table_from_file(
            self.vocabulary_file,
            vocab_size=self.vocabulary_size - 1,
            num_oov_buckets=1)

    def _process(self, data, input_data):
        length = tf.shape(data['labels'])[0]
        ids = self.vocabulary.lookup(data['labels'])
        data['ids'] = ids
        data['length'] = length
        del data['labels']
        return data

    def batch(self, dataset, batch_size):
        return dataset.padded_batch(
            batch_size, get_padded_shapes(dataset))

    def transform(self, inputs, mode):
        ids, _ = inputs
        try:
            embeddings = tf.get_variable(
                "t_embs", dtype=self.dtype, trainable=self.trainable)
        except ValueError:
            shape = [self.vocabulary_size, self.embedding_size]
            embeddings = tf.get_variable(
                "t_embs",
                shape=shape,
                dtype=self.dtype,
                trainable=self.trainable)

        embeddings = tf.nn.embedding_lookup(embeddings, ids)
        if (isinstance(self.dropout_rate, tf.Tensor) or
                self.dropout_rate > 0 and mode == tf.estimator.ModeKeys.TRAIN):
            embeddings = tf.layers.dropout(
                embeddings,
                rate=self.dropout_rate)

        return embeddings

    def get_example_size(self, example):
        return example['length']


class SubtokenEmbedder(TokenEmbedder):
    def __init__(self,
                 subtokenizer: Callable,
                 vocabulary_file_key,
                 embedding_size: int,
                 dropout_rate: Union[int, tf.Tensor] = 0.0,
                 lowercase=True,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32):
        super().__init__(
            vocabulary_file_key,
            embedding_size,
            dropout_rate,
            lowercase,
            trainable,
            dtype)
        self.subtokenizer = subtokenizer

    def extract_tensors(self):
        #TODO: Truncation
        def _tensor_extractor(sample):
            indices, labels = [], []
            for i, token in enumerate(sample):
                for subtoken in self.subtokenizer(token):
                    indices.append(i)
                    labels.append(subtoken.lower() if self.lowercase else subtoken)
            return {"indices": indices, "labels": labels, "length": len(sample)}

        tensor_types = {
            "length": tf.int32,
            "labels": tf.string,
            "indices": tf.int32
        }
        tensor_shapes = {
            "length": tf.TensorShape([]),
            "labels": tf.TensorShape([None]),
            "indices": tf.TensorShape([None])
        }
        return _tensor_extractor, tensor_types, tensor_shapes

    def _process(self, data: Dict[str, tf.Tensor], input_data)-> Dict[str, tf.Tensor]:
        indices = tf.cast(tf.expand_dims(data['indices'], 1), tf.int64)

        ids = tf.cast(self.vocabulary.lookup(data['labels']), tf.int64)
        ids = tf.SparseTensor(indices, ids, (tf.cast(data['length'], tf.int64),))

        data['ids'] = ids
        del data['indices']
        del data['labels']
        return data

    def batch(self, dataset, batch_size):
        batch_fn_map = {"features": shifted_batch}
        return diverse_batch(dataset, batch_size, batch_fn_map)

    def transform(self, inputs, mode):
        ids, lengths = inputs
        try:
            embeddings = tf.get_variable(
                "t_embs", dtype=self.dtype, trainable=self.trainable)
        except ValueError:
            shape = [self.vocabulary_size, self.embedding_size]
            embeddings = tf.get_variable(
                "t_embs",
                shape=shape,
                dtype=self.dtype,
                trainable=self.trainable)

        max_vertices = tf.reduce_max(lengths)
        features = tf.nn.embedding_lookup_sparse(
            embeddings, ids, None,
            combiner="mean")
        return tf.reshape(features, (-1, max_vertices, self.embedding_size))
