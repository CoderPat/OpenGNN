from typing import Union, List, Tuple, Callable

import tensorflow as tf
import numpy as np

from opengnn.inputters.token_embedder import TokenEmbedder, SubtokenEmbedder
from opengnn.utils.data import shifted_batch, get_padded_shapes, diverse_batch
from opengnn.utils.misc import count_lines


def extended_lookup(tokens: List[str],
                    original_ids: List[int],
                    extension_tokens: List[str],
                    extension_original_ids: List[int],
                    vocabulary_size: int,
                    unk_token_id: int) -> Tuple[List[int], List[str]]:
    """
    Does a lookup on the "extended" vocabulary originating from a the extension
    of an original vocabulary plus unseen tokens on a sequence of tokens that extend
    the vocabulary

    Args:
        tokens: the tokens that we are doing a lookup for
        original_ids: the ids assigned to the tokens on the original vocabulary
        extension_tokens: a list of tokens that extends the original vocabulary
        extension_original_ids: the ids assigned to extension_tokens on the original vocabulary
        vocabulary_size: the total size of the original vocabulary
        unk_token_id: the id of the <unk> token in the original vocabulary
    Returns:
        The ids for the tokens on the extended vocabulary plus a mappings for ids out of the
            original vocabulary to tokens
    """
    # firstly extend the vocabulary using the extension_tokens and their ids
    extended_vocab = {}
    pointer_map = []
    for token, idx in zip(extension_tokens, extension_original_ids):
        # If a extension token is not in the original vocabulary and it was not seen yet we add
        # it to the extended vocabulary and create the mapping for this new id to string
        if idx == unk_token_id:
            if token not in extended_vocab:
                pointer_map.append(token)
                extended_vocab[token] = vocabulary_size + len(extended_vocab)

    # with the extended vocabulary created we now can perform the extended lookup
    corrected_ids = []
    for token, idx in zip(tokens, original_ids):
        if idx == unk_token_id and token in extended_vocab:
            corrected_ids.append(extended_vocab[token])
        else:
            corrected_ids.append(idx)

    return np.array(corrected_ids, dtype=np.int64), np.array(pointer_map, dtype=object)


class CopyingTokenEmbedder(TokenEmbedder):
    def __init__(self,
                 vocabulary_file_key,
                 embedding_size: int,
                 output_vocabulary_file_key=None,
                 input_tokens_fn=None,
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
        super().__init__(
            vocabulary_file_key=vocabulary_file_key,
            embedding_size=embedding_size,
            dropout_rate=dropout_rate,
            truncated_sentence_size=truncated_sentence_size,
            trainable=trainable,
            lowercase=lowercase,
            dtype=dtype)
        self.output_vocabulary_file_key = output_vocabulary_file_key
        self.input_tokens_fn = input_tokens_fn

    def initialize(self, metadata):
        super().initialize(metadata)
        if self.output_vocabulary_file_key is not None:
            # initialize output vocabulary
            # it's the most encapsulated way I see of doing this
            self.output_vocabulary_file = metadata[self.output_vocabulary_file_key]

            self.output_vocabulary_size = \
                count_lines(self.output_vocabulary_file) + 1
            self.output_vocabulary = tf.contrib.lookup.index_table_from_file(
                self.output_vocabulary_file,
                vocab_size=self.output_vocabulary_size - 1,
                num_oov_buckets=1)

    def _process(self, data, input_data):
        if input_data is None:
            # if no input data parameter is passed, we can assume this is an inputter for input data
            if self.output_vocabulary_file_key is None:
                raise ValueError(
                    "CopyingTokenEmbedders needs output vocabulary when processing input data")

            # do this outside pyfunc since its probably (much) faster
            out_ids = self.output_vocabulary.lookup(data['labels'])

            # perform the extend lookup on tokens using themselves as extensions
            data['out_ids'], data['pointer_map'] = tf.py_func(
                lambda labels, out_ids:
                    extended_lookup(
                        labels, out_ids, labels, out_ids,
                        self.output_vocabulary_size, self.output_vocabulary_size - 1),
                inp=(data['labels'], out_ids),
                Tout=(tf.int64, tf.string))
            data['out_ids'].set_shape([None])
            data['pointer_map'].set_shape([None])
            data['ids'] = self.vocabulary.lookup(data['labels'])
            data['length'] = tf.shape(data['labels'])[0]

        else:
            # this is output data
            if self.input_tokens_fn is None:
                raise ValueError(
                    "CopyingTokenEmbedders needs input_token_fn when processing output data")

            ids = self.vocabulary.lookup(data['labels'])
            input_tokens = self.input_tokens_fn(input_data)
            if self.lowercase:
                input_tokens = tf.py_func(
                    lambda tokens: np.array([token.lower() for token in tokens], dtype=object),
                    [input_tokens], tf.string, stateful=False)

            input_out_ids = self.vocabulary.lookup(input_tokens)

            # perform lookup using the input tokens as extensions
            data['ids'] = tf.py_func(
                lambda labels, ids, input_labels, input_ids:
                    extended_lookup(
                        labels, ids, input_labels, input_ids,
                        self.vocabulary_size, self.vocabulary_size - 1)[0],
                inp=(data['labels'], ids, input_tokens, input_out_ids),
                Tout=tf.int64)
            data['ids'].set_shape([None])
            data['length'] = tf.shape(data['labels'])[0]

        # delete no longer necessary labels
        del data['labels']
        return data

    def batch(self, dataset, batch_size):
        return dataset.padded_batch(
            batch_size, get_padded_shapes(dataset))

    def transform(self, inputs, mode):
        ids, length = inputs

        # clip pointers to unk
        clipped_ids = tf.clip_by_value(ids, 0, self.vocabulary_size - 1)
        return super().transform((clipped_ids, length), mode)


class CopyingSubtokenEmbedder(CopyingTokenEmbedder, SubtokenEmbedder):
    def __init__(self,
                 subtokenizer: Callable,
                 vocabulary_file_key,
                 embedding_size: int,
                 output_vocabulary_file_key=None,
                 input_tokens_fn=None,
                 dropout_rate: Union[int, tf.Tensor] = 0.0,
                 truncated_sentence_size=None,
                 lowercase=True,
                 trainable: bool = True,
                 dtype: tf.DType = tf.float32):
        SubtokenEmbedder.__init__(
            self,
            subtokenizer,
            vocabulary_file_key,
            embedding_size,
            dropout_rate,
            lowercase,
            trainable,
            dtype)
        self.output_vocabulary_file_key = output_vocabulary_file_key
        self.input_tokens_fn = input_tokens_fn

    def extract_tensors(self):
        def _tensor_extractor(sample):
            indices, labels, full_labels = [], [], []
            for i, token in enumerate(sample):
                full_labels.append(token.lower() if self.lowercase else token)
                for subtoken in self.subtokenizer(token):
                    indices.append(i)
                    labels.append(subtoken.lower() if self.lowercase else subtoken)
            return {"indices": indices, "labels": labels, "full_labels": full_labels, "length": len(sample)}

        tensor_types = {
            "length": tf.int32,
            "labels": tf.string,
            "full_labels": tf.string,
            "indices": tf.int32
        }
        tensor_shapes = {
            "length": tf.TensorShape([]),
            "labels": tf.TensorShape([None]),
            "full_labels": tf.TensorShape([None]),
            "indices": tf.TensorShape([None])
        }
        return _tensor_extractor, tensor_types, tensor_shapes

    def initialize(self, metadata):
        return CopyingTokenEmbedder.initialize(self, metadata)

    def _process(self, data, input_data):
        if input_data is None:
            # if no input data parameter is passed, we can assume this is an inputter for input data
            if self.output_vocabulary_file_key is None:
                raise ValueError(
                    "CopyingTokenEmbedders needs output vocabulary when processing input data")

            # do this outside pyfunc since its probably (much) faster
            out_ids = self.output_vocabulary.lookup(data['full_labels'])

            # perform the extend lookup on tokens using themselves as extensions
            data['out_ids'], data['pointer_map'] = tf.py_func(
                lambda labels, out_ids:
                    extended_lookup(
                        labels, out_ids, labels, out_ids,
                        self.output_vocabulary_size, self.output_vocabulary_size - 1),
                inp=(data['full_labels'], out_ids),
                Tout=(tf.int64, tf.string))
            data['out_ids'].set_shape([None])
            data['pointer_map'].set_shape([None])
            indices = tf.cast(tf.expand_dims(data['indices'], 1), tf.int64)
            ids = tf.cast(self.vocabulary.lookup(data['labels']), tf.int64)
            ids = tf.SparseTensor(indices, ids, (tf.cast(data['length'], tf.int64),))

            data['ids'] = ids

        else:
            # this is output data
            if self.input_tokens_fn is None:
                raise ValueError(
                    "CopyingSubtokenEmbedders needs input_token_fn when processing output data")

            ids = self.vocabulary.lookup(data['labels'])
            input_tokens = self.input_tokens_fn(input_data)
            if self.lowercase:
                input_tokens = tf.py_func(
                    lambda tokens: np.array([token.lower() for token in tokens], dtype=object),
                    [input_tokens], tf.string, stateful=False)

            input_out_ids = self.vocabulary.lookup(input_tokens)

            indices = tf.cast(tf.expand_dims(data['indices'], 1), tf.int64)
            # perform lookup using the input tokens as extensions
            data['ids'] = tf.py_func(
                lambda labels, ids, input_labels, input_ids:
                    extended_lookup(
                        labels, ids, input_labels, input_ids,
                        self.vocabulary_size, self.vocabulary_size - 1)[0],
                inp=(data['full_labels'], ids, input_tokens, input_out_ids),
                Tout=tf.int64)
            data['ids'].set_shape([None])
            data['ids'] = tf.SparseTensor(indices, ids, (tf.cast(data['length'], tf.int64),))

        # delete no longer necessary labels
        del data['indices']
        del data['labels']
        del data['full_labels']
        return data

    def batch(self, dataset, batch_size):
        batch_fn_map = {"features": shifted_batch}
        return diverse_batch(
            dataset, batch_size, batch_fn_map,
            default_batch_fn=lambda dataset, batch_size: dataset.padded_batch(batch_size, get_padded_shapes(dataset)))

    def transform(self, inputs, mode):
        ids, length = inputs

        # clip pointers to unk
        # TODO: clip pointers inside SparseTensor since clip_by_value only accepts Tensors
        # currently it doesn't matter since we dont use this inputter for target, where clipping is important
        return SubtokenEmbedder.transform(self, (ids, length), mode)
