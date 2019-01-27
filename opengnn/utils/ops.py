import tensorflow as tf

import opengnn.constants as constants


def batch_unsrt_segment_sum(data, segment_ids, num_segments):
    """ Performas the `tf.unsorted_segment_sum` operation batch-wise"""
    # create distinct segments per batch
    num_batches = tf.shape(segment_ids, out_type=tf.int64)[0]
    batch_indices = tf.range(num_batches)
    segment_ids_per_batch = segment_ids + num_segments * tf.expand_dims(batch_indices, axis=1)

    # do the normal unsegment sum and reshape to original shape
    seg_sums = tf.unsorted_segment_sum(data, segment_ids_per_batch, num_segments * num_batches)
    return tf.reshape(seg_sums, tf.stack((-1, num_segments)))


def batch_unsrt_segment_max(data, segment_ids, num_segments):
    """ Performas the `tf.unsorted_segment_max` operation batch-wise"""
    # create distinct segments per batch
    num_batches = tf.shape(segment_ids, out_type=tf.int64)[0]
    batch_indices = tf.range(num_batches)
    segment_ids_per_batch = segment_ids + num_segments * tf.expand_dims(batch_indices, axis=1)

    # do the normal unsegment sum and reshape to original shape
    seg_maxs = tf.unsorted_segment_max(data, segment_ids_per_batch, num_segments * num_batches)
    return tf.reshape(seg_maxs, tf.stack((-1, num_segments)))


def batch_unsrt_segment_logsumexp(data, segment_ids, num_segments):
    """ Adds probabilities in log-space for each segment in a numerically stable way """
    # extract max for each segment and regather
    params_max = batch_unsrt_segment_max(data, segment_ids, num_segments)
    data_max = batch_gather(params_max, segment_ids)

    # subtract maxes from each element in data, exponentiante and add the probabilies
    data = data - data_max
    data = tf.exp(data)
    params = batch_unsrt_segment_sum(data, segment_ids, num_segments)

    # transform back to log-space and add the max back
    params = tf.log(params + constants.SMALL_NUMBER)
    return params + params_max


def batch_gather(embeddings, indices):
    """ Performs the `tf.gather` operation batch-wise """
    # flatten embeddings in the batch dimension
    shape = tf.shape(embeddings, out_type=indices.dtype)
    embeddings_f = tf.reshape(embeddings, tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0))

    # Add offsets to indices so as to collect the correct embedding
    offset_shape = tf.concat(
        [[shape[0]], 
         tf.cast([1 for _ in range(indices.shape.ndims - 1)], dtype=indices.dtype)], axis=0)
    offset = tf.reshape(tf.range(shape[0]) * shape[1], offset_shape)
    
    # do the normal gather
    output = tf.gather(embeddings_f, indices + offset)
    return output


def stack_indices(tensor, axis):
    """
    Given a tensor with D dimensions, this function returns a tensor with D+1 dimensions where
    where the values are the original value and the index in a given axis (with the index first)
    """
    # get indices for the given axis using range
    indices = tf.range(tf.shape(tensor, out_type=tensor.dtype)[axis], dtype=tensor.dtype)
    
    # expand dims up to the rank of the original tensor
    # TODO: there must be a better way to do this, but couldn't find in expand_dims docs
    for _ in range(axis):
        indices = tf.expand_dims(indices, 0)
    for _ in range(axis+1, len(tensor.shape)):
        indices = tf.expand_dims(indices, -1)
    
    # tile the indices in all dimensions except the on the axis
    tile_shape = tf.concat([tf.shape(tensor)[:axis], [1], tf.shape(tensor)[axis+1:]], axis=0)
    return tf.stack([tf.tile(indices, tile_shape), tensor], axis=-1)