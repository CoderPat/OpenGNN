from abc import ABC, abstractmethod
from typing import Dict, Any

import tensorflow as tf
import numpy as np

from opengnn.utils.data import diverse_batch, batch_and_bucket_by_size
from opengnn.utils.data import filter_examples_by_size, truncate_examples_by_size


def optimize(loss: tf.Tensor, params: Dict[str, Any]):
    global_step = tf.train.get_or_create_global_step()

    optimizer = params.get('optimizer', 'Adam')
    if optimizer != 'Adam':
        optimizer_class = getattr(tf.train, optimizer, None)
        if optimizer_class is None:
            raise ValueError("Unsupported optimizer %s" % optimizer)

        optimizer_params = params.get("optimizer_params", {})

        def optimizer(lr): return optimizer_class(lr, **optimizer_params)

    learning_rate = params['learning_rate']
    if params.get('decay_rate') is not None:
        learning_rate = tf.train.exponential_decay(
            learning_rate,
            global_step,
            decay_steps=params.get('decay_steps', 1),
            decay_rate=params['decay_rate'],
            staircase=True)

    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=global_step,
        learning_rate=learning_rate,
        clip_gradients=params['clip_gradients'],
        summaries=[
            "learning_rate",
            "global_gradient_norm",
        ],
        optimizer=optimizer,
        name="optimizer")


class Model(ABC):
    def __init__(self,
                 name: str,
                 features_inputter=None,
                 labels_inputter=None) -> None:
        self.name = name
        self.features_inputter = features_inputter
        self.labels_inputter = labels_inputter

    def model_fn(self):
        def _model_fn(features, labels, mode, params, config=None):
            if mode == tf.estimator.ModeKeys.TRAIN:
                with tf.variable_scope(self.name):
                    # build models graph
                    outputs, predictions = self.__call__(
                        features, labels, mode, params, config)
                    # compute loss, tb_loss and train_op
                    loss, tb_loss = self.compute_loss(
                        features, labels, outputs, params, mode)

                train_op = optimize(loss, params)

                return tf.estimator.EstimatorSpec(
                    mode, loss=tb_loss, train_op=train_op)

            elif mode == tf.estimator.ModeKeys.EVAL:
                with tf.variable_scope(self.name):
                    # build models graph
                    outputs, predictions = self.__call__(
                        features, labels, mode, params, config)
                    # compute loss, tb_loss and metric ops
                    loss, tb_loss = self.compute_loss(
                        features, labels, outputs, params, mode)

                metrics = self.compute_metrics(
                    features, labels, predictions)

                # TODO: this assumes that the loss across validation can be
                # calculated as the average over the loss of the minibatch
                # which is not always the case (cross entropy averaged over time an batch)
                # but if minibatch a correctly shuffled, this is a good aproximation for now
                return tf.estimator.EstimatorSpec(
                    mode, loss=tb_loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope(self.name):
                    # build models graph
                    _, predictions = self.__call__(
                        features, labels, mode, params, config)
                return tf.estimator.EstimatorSpec(
                    mode, predictions=predictions)

        return _model_fn

    def input_fn(self,
                 mode: tf.estimator.ModeKeys,
                 batch_size: int,
                 metadata,
                 features_file,
                 labels_file=None,
                 sample_buffer_size=None,
                 maximum_features_size=None,
                 maximum_labels_size=None,
                 features_bucket_width=None,
                 labels_bucket_width=None,
                 num_threads=None):
        assert not (mode != tf.estimator.ModeKeys.PREDICT and
                    labels_file is None)

        # the function returned
        def _input_fn():
            self.initialize(metadata)
            feat_dataset, feat_process_fn, feat_batch_fn, features_size_fn =\
                self.get_features_builder(features_file, mode)

            if labels_file is not None:
                labels_dataset, labels_process_fn, \
                    labels_batch_fn, labels_size_fn = \
                    self.get_labels_builder(labels_file, mode)
                dataset = tf.data.Dataset.zip((feat_dataset, labels_dataset))

                def process_fn(features, labels):
                    return feat_process_fn(features), labels_process_fn(labels, features)

                def batch_fn(dataset, batch_size):
                    return diverse_batch(
                        dataset, batch_size,
                        (feat_batch_fn, labels_batch_fn))

                example_size_fns = [features_size_fn, labels_size_fn]
                bucket_widths = [features_bucket_width, labels_bucket_width]
                maximum_example_size = (maximum_features_size, maximum_labels_size)

            else:
                dataset = feat_dataset
                process_fn = feat_process_fn
                batch_fn = feat_batch_fn
                example_size_fns = features_size_fn
                bucket_widths = features_bucket_width
                maximum_example_size = maximum_features_size

            # shuffle, process batch and allow repetition
            # TODO: Fix derived seed (bug in tensorflow)
            seed = np.random.randint(np.iinfo(np.int64).max)
            if sample_buffer_size is not None:
                dataset = dataset.shuffle(
                    sample_buffer_size,
                    reshuffle_each_iteration=False,
                    seed=seed)
            dataset = dataset.map(process_fn, num_parallel_calls=num_threads or 4)
            dataset = dataset.apply(filter_examples_by_size(
                example_size_fns=example_size_fns,
                maximum_example_sizes=maximum_example_size))

            dataset = dataset.apply(batch_and_bucket_by_size(
                batch_size=batch_size,
                batch_fn=batch_fn,
                bucket_widths=bucket_widths,
                example_size_fns=example_size_fns))

            if mode == tf.estimator.ModeKeys.TRAIN:
                dataset = dataset.repeat()

            return dataset.prefetch(None)

        return _input_fn

    def initialize(self, metadata):
        """
        Runs model specific initialization (e.g. vocabularies loading).

        Args:
            metadata: A dictionary containing additional metadata set
                by the user.
        """
        if self.features_inputter is not None:
            self.features_inputter.initialize(metadata)
        if self.labels_inputter is not None:
            self.labels_inputter.initialize(metadata)

    @abstractmethod
    def __call__(self, features, labels, mode, params, config=None):
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, features, labels, outputs, params, mode):
        raise NotImplementedError()

    @abstractmethod
    def compute_metrics(self, features, labels, predictions):
        raise NotImplementedError()

    def get_features_builder(self, features_file, mode):
        if self.features_inputter is None:
            raise NotImplementedError()

        dataset = self.features_inputter.make_dataset(features_file, mode)
        process_fn = self.features_inputter.process
        batch_fn = self.features_inputter.batch
        size_fn = self.features_inputter.get_example_size
        return dataset, process_fn, batch_fn, size_fn

    def get_labels_builder(self, labels_file, mode):
        if self.labels_inputter is None:
            raise NotImplementedError()

        dataset = self.labels_inputter.make_dataset(labels_file, mode)
        process_fn = self.labels_inputter.process
        batch_fn = self.labels_inputter.batch
        size_fn = self.labels_inputter.get_example_size
        return dataset, process_fn, batch_fn, size_fn
