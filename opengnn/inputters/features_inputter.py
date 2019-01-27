import tensorflow as tf

from opengnn.inputters.inputter import Inputter
from opengnn.utils.misc import find_max


class FeaturesInputter(Inputter):
    def extract_metadata(self, data_filepath: str) -> None:
        """
        """
        self.features_size = find_max(
            data_filepath,
            lambda sample: len(sample))

    def extract_tensors(self):
        def _tensor_extractor(sample):
            return {"features": sample}

        tensor_types = {"features": tf.float32}
        tensor_shapes = {"features": tf.TensorShape([self.features_size])}
        return _tensor_extractor, tensor_types, tensor_shapes

    def _process(self, data, input_data):
        return data

    def batch(self, dataset, batch_size):
        return dataset.batch(batch_size)

    def transform(self, inputs, lengths=None):
        return inputs

    def get_example_size(self, example):
        return 1
