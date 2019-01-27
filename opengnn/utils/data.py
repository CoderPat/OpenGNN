import tensorflow as tf
from collections import Iterable


def get_padded_shapes(dataset):
    """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.
    Args:
      dataset: The dataset that will be batched with padding.
    Returns:
      The same structure as ``dataset.output_shapes`` containing the padded
      shapes.
    """
    return tf.contrib.framework.nest.map_structure(
        lambda shape: shape.as_list(), dataset.output_shapes)


def filter_examples_by_size(maximum_example_sizes=None,
                            example_size_fns=None):
    """Transformation that constrains examples length.

    Returns:
    A ``tf.data.Dataset`` transformation.
    """
    if example_size_fns is None:
        return lambda dataset: dataset

    if not isinstance(example_size_fns, Iterable):
        example_size_fns = [example_size_fns]

    if not isinstance(maximum_example_sizes, Iterable):
        maximum_example_sizes = [maximum_example_sizes]

    def _predicate(*args):
        constraints = []
        for i, (example_size_fn, maximum_length) in \
                enumerate(zip(example_size_fns, maximum_example_sizes)):
            if maximum_length is not None:
                features_size = example_size_fn(args[i])
                constraints.append(tf.less_equal(features_size, maximum_length))
        return tf.reduce_all(constraints)

    return lambda dataset: dataset.filter(_predicate)


def truncate_examples_by_size(maximum_truncated_sizes=None,
                              truncation_fns=None):
    """Transformation that truncates examples length.

    Returns:
    A ``tf.data.Dataset`` transformation.
    """
    if truncation_fns is None:
        return lambda dataset: dataset

    if not isinstance(truncation_fns, Iterable):
        truncation_fns = [truncation_fns]

    if not isinstance(maximum_truncated_sizes, Iterable):
        maximum_truncated_sizes = [maximum_truncated_sizes]

    def _transform(*args):
        transformed_args = []
        for i, (truncation_fn, maximum_length) in \
                enumerate(zip(truncation_fns, maximum_truncated_sizes)):
            if maximum_length is not None:
                transformed_args.append(truncation_fn(args[i]))
            else:
                transformed_args.append(args[i])

        return tuple(transformed_args)

    return lambda dataset: dataset.map(_transform)


def batch_and_bucket_by_size(batch_size=None,
                             batch_fn=None,
                             bucket_widths=None,
                             example_size_fns=None,):

    if bucket_widths is None or \
            isinstance(bucket_widths, Iterable) and all(width is None for width in bucket_widths):
        return lambda dataset: batch_fn(dataset, batch_size)

    if not isinstance(bucket_widths, Iterable):
        bucket_widths = [bucket_widths]

    if not isinstance(example_size_fns, Iterable):
        example_size_fns = [example_size_fns]

    def _key(*args):
        bucket_ids = []
        for i, (bucket_width, example_size_fn) in \
                enumerate(zip(bucket_widths, example_size_fns)):
            if bucket_width is not None:
                features_size = example_size_fn(args[i])
                bucket_ids.append(features_size // bucket_width)
        if len(bucket_ids) == 1:
            return tf.cast(bucket_ids[0], tf.int64)
        else:
            # TODO: implement some bijective pairing function, like cantor's
            pass

    def _reduce(key, dataset):
        return batch_fn(dataset, batch_size)

    return lambda dataset: dataset.apply(tf.contrib.data.group_by_window(
        _key,
        _reduce,
        window_size=batch_size))


def prune_dataset(dataset, prune_keys):
    """
    Args:
        dataset ([type]): [description]
        prune_keys ([type]): [description]

    Returns:
        [type]: [description]
    """
    def _prune_by_keys(data):
        pruned_data = {}
        for key, value in data.items():
            if key not in prune_keys:
                pruned_data[key] = value
        return pruned_data

    return dataset.map(lambda data: _prune_by_keys(data))


def merge_datasets(datasets):
    """
    Args:
        datasets ([type]): [description]

    Returns:
        [type]: [description]
    """
    datasets = tf.data.Dataset.zip(datasets)

    def _merge_data(*datas):
        merged_data = {}
        for data in datas:
            for key, value in data.items():
                merged_data[key] = value
        return merged_data

    return datasets.map(_merge_data)


def shifted_batch(dataset, batch_size):
    """
    Args:
        dataset ([type]): [description]
        batch_size ([type]): [description]
        key ([type]): [description]

    Returns:
        [type]: [description]
    """
    def _shift_indices(data):
        if isinstance(data, dict):
            for key, value in data.items():
                data[key] = _shift_indices(value)
            return data

        batch_size = tf.shape(data, out_type=tf.int64)[0]
        indices = data.indices
        max_tokens = tf.shape(data, out_type=tf.int64)[1]
        new_indices = indices[:, 0]*max_tokens + indices[:, 1]

        # TODO: Fix padding in the middle of the sparse matrix
        # (currently only supports zero embeddings for
        # padding for middle padding)
        last_token = tf.reduce_max(new_indices) + 1
        max_padded_token = batch_size*max_tokens

        new_indices = tf.concat([
            new_indices,
            tf.range(last_token, max_padded_token)],
            axis=0)
        new_values = tf.concat([
            data.values,
            tf.zeros((tf.cast(max_padded_token - last_token, tf.int32),),
                     tf.int64)],
            axis=0)

        data = tf.SparseTensor(
            tf.expand_dims(new_indices, 1), new_values, (max_padded_token,))
        return data

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_shift_indices)
    return dataset


def diverse_batch(dataset,
                  batch_size,
                  batch_fns,
                  default_batch_fn=None):
    """
    Args:
        dataset ([type]): [description]
        batch_size ([type]): [description]
        batch_fn_maps ([type]): [description]
        default_batch_fn ([type], optional): Defaults to None. [description]

    Returns:
        [type]: [description]

    """
    if default_batch_fn is None:
        def default_batch_fn(dataset, batch_size):  # pylint: disable=E0102
            return dataset.batch(batch_size)

    if isinstance(batch_fns, dict):
        return _dict_diverse_batch(
            dataset, batch_size, batch_fns, default_batch_fn)
    else:
        return _zip_diverse_batch(dataset, batch_size, batch_fns)


def _dict_diverse_batch(dataset,
                        batch_size,
                        batch_fn_maps,
                        default_batch_fn):

    # do specific batching strategy for each key tensor dataset in question
    batched_datasets = []
    keyset = set()
    for key, batch_fn in batch_fn_maps.items():
        if isinstance(key, tuple):
            def key_fn(data): return {k: data[k] for k in key}
            keyset = keyset.union(set(key))
        else:
            def key_fn(data): return {key: data[key]}
            keyset.add(key)

        keyed_dataset = dataset.map(key_fn)
        batched_datasets.append(batch_fn(keyed_dataset, batch_size))

    # apply default batching on the rest
    pruned_dataset = prune_dataset(dataset, keyset)
    batched_datasets.append(default_batch_fn(pruned_dataset, batch_size))
    return merge_datasets(tuple(batched_datasets))


def _zip_diverse_batch(dataset,
                       batch_size,
                       batch_fns):

    batched_datasets = []
    for i, batch_fn in enumerate(batch_fns):
        indexed_dataset = dataset.map(lambda *args: args[i])
        batched_datasets.append(batch_fn(indexed_dataset, batch_size))
    return tf.data.Dataset.zip(tuple(batched_datasets))
