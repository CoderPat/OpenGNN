from abc import ABC, abstractmethod
import json

import tensorflow as tf

from opengnn.utils.misc import read_file


class Inputter(ABC):
    def __init__(self):
        self.process_hooks = []

    def initialize(self, metadata):
        """[summary]
        """
        pass

    def add_process_hooks(self, hooks):
        """Adds processing hooks.
        Processing hooks are additional and model specific data processing
        functions applied after calling this inputter
        :meth:`opennmt.inputters.inputter.Inputter.process` function.
        Args:
        hooks: A list of callables with the signature
            ``(inputter, data) -> data``.
        """
        self.process_hooks.extend(hooks)

    def make_dataset(self, data_filename: str, mode: tf.estimator.ModeKeys)-> tf.data.Dataset:
        """Creates the dataset required by this inputter.
        Args:
            data_filename: The data file.
        Returns:
            A ``tf.data.Dataset``.
        """
        # Extract general information for the model using (a file from) the dataset
        # TODO: I dont find this "peek at the file" solution very elegant.
        self.extract_metadata(data_filename)

        extractor, tensor_types, tensor_shapes = self.extract_tensors()

        # TODO: switch caching from here to Dataset API
        # might bring problems in the future if files too big
        def _generator():
            with read_file(data_filename) as f:
                for line in f:
                    sample = json.loads(line)
                    yield extractor(sample)

        return tf.data.Dataset.from_generator(
            _generator, tensor_types, tensor_shapes)

    def extract_metadata(self, data_file):
        """[summary]

        Args:
            data_file ([type]): [description]
        """
        pass

    @abstractmethod
    def extract_tensors(self):
        """[summary]

        Args:
            data_file ([type]): [description]
        """
        raise NotImplementedError()

    def process(self, data, input_data=None):
        """Prepares raw data.
        Args:
            data: The raw data.
            input_data: The raw input data (in case we are processing output data).
        Returns:
            A dictionary of ``tf.Tensor``.
        """
        # shallow-copy the data since other inputters might use the data
        data = self._process(dict(data), input_data)
        for hook in self.process_hooks:
            data = hook(self, data)
        return data

    @abstractmethod
    def _process(self, data):
        """
        Args:
            data ([type]): [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def batch(self, dataset, batch_size):
        """
        Args:
            dataset ([type]): [description]
            batch_size ([type]): [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def transform(self, inputs, mode):
        """Transforms inputs.
        Args:
            inputs: A (possible nested structure of) ``tf.Tensor`` which
                depends on the inputter.
            mode: A ``tf.estimator.ModeKeys`` mode.
        Returns:
            The transformed input.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_example_size(self, example):
        """
        """
        raise NotImplementedError()
