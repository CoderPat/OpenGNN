# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Any

try:
    from typing import Counter as CounterType
    from typing import Tuple
    CounterTupleType = CounterType[Tuple]
except ImportError:
    # typing.Counter was only introduced in python 3.6.1.
    CounterTupleType = Any


import collections
import math

import numpy as np

import tensorflow as tf

from opengnn.utils.misc import find


def _get_ngrams(segment: List[Any], max_order: int, min_order: int=1) -> CounterTupleType:
    """Extracts all n-grams upto a given maximum order from an input segment.

    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.

    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """

    ngram_counts = collections.Counter()  # type: CounterTupleType
    for order in range(min_order, max_order + 1):
        for i in range(len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus,
                 translation_corpus,
                 end_token: Any=None,
                 max_order: int=4,
                 use_bp: bool=True) -> float:
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.

    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for (reference, translation) in zip(reference_corpus, translation_corpus):
        if end_token is not None:
            reference = reference[:find(reference, end_token)]
            translation = translation[:find(translation, end_token)]
        reference_length += len(reference)
        translation_length += len(translation)
        ref_ngram_counts = _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(
                ngram)-1] += translation_ngram_counts[ngram]

    precisions = [0] * max_order  # type: List[float]
    for i in range(max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum/max_order)

    if use_bp:
        if not reference_length:
            bp = 1.0
        else:
            ratio = translation_length / reference_length
            if ratio <= 0.0:
                bp = 0.0
            elif ratio >= 1.0:
                bp = 1.0
            else:
                bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return np.float32(bleu)


def compute_rouge(reference_corpus,
                  translation_corpus,
                  end_token=None,
                  max_order=2) -> float:
    """Computes ROUGE-N f1 score of two text collections of sentences.

    Source: https://www.microsoft.com/en-us/research/publication/
      rouge-a-package-for-automatic-evaluation-of-summaries/

    Args:
        eval_sentences: The sentences that have been picked by the summarizer
        ref_sentences: The sentences from the reference set
        n: Size of ngram.  Defaults to 2.

    Returns:
        f1 score for ROUGE-N
    """
    f1_scores = []
    for (reference, translation) in zip(reference_corpus, translation_corpus):
        if end_token is not None:
            reference = reference[:find(reference, end_token)]
            translation = translation[:find(translation, end_token)]

        eval_ngrams = set(_get_ngrams(translation, max_order, max_order).keys())
        ref_ngrams = set(_get_ngrams(reference, max_order, max_order).keys())
        ref_count = len(ref_ngrams)
        eval_count = len(eval_ngrams)

        # Gets the overlapping ngrams between evaluated and reference
        overlapping_ngrams = eval_ngrams.intersection(ref_ngrams)
        overlapping_count = len(overlapping_ngrams)

        # Handle edge case. This isn't mathematically correct, but it's good enough
        if eval_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / eval_count

        if ref_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / ref_count

        f1_scores.append(2.0 * ((precision * recall) / (precision + recall + 1e-8)))

    # return overlapping_count / reference_count
    return np.mean(f1_scores, dtype=np.float32)


def compute_f1(references, translations, end_token=None, beta=1):
    """
    Computes BLEU for a evaluation set of translations
    Based on https://github.com/mast-group/convolutional-attention/blob/master/convolutional_attention/f1_evaluator.pyy
    """

    total_f1 = 0
    for (reference, translation) in zip(references, translations):
        if end_token is not None:
            reference = reference[:find(reference, end_token)]
            translation = translation[:find(translation, end_token)]

        tp = 0
        ref = list(reference)
        for token in set(translation):
            if token in ref:
                ref.remove(token)
                tp += 1

        if len(translation) > 0:
            precision = tp / len(translation)
        else:
            precision = 0

        if len(reference) > 0:
            recall = tp / len(reference)
        else:
            recall = 0

        if precision + recall > 0:
            f1 = (1+beta**2) * precision * recall / ((beta**2 * precision) + recall)
        else:
            f1 = 0

        total_f1 += f1
    return np.float32(total_f1 / len(translations))


def bleu_score(labels, predictions, end_token, **unused_kwargs):
    """BLEU score computation between labels and predictions.

    An approximate BLEU scoring method since we do not glue word pieces or
    decode the ids and tokenize the output. By default, we use ngram order of 4
    and use brevity penalty. Also, this does not have beam search.

    Args:
      predictions: tensor, model predicitons
      labels: tensor, gold output.

    Returns:
      bleu: int, approx bleu score
    """
    bleu = tf.py_func(compute_bleu, (labels, predictions, end_token), tf.float32)
    return tf.metrics.mean(bleu)


def rouge_2_fscore(labels, prediction, end_token, **unused_kwargs):
    """ROUGE-2 F1 score computation between labels and predictions.

    This is an approximate ROUGE scoring method since we do not glue word pieces
    or decode the ids and tokenize the output.

    Args:
        predictions: tensor, model predictions
        labels: tensor, gold output.

    Returns:
        rouge2_fscore: approx rouge-2 f1 score.
    """
    rouge_2_f_score = tf.py_func(compute_rouge, (labels, prediction, end_token), tf.float32)
    return tf.metrics.mean(rouge_2_f_score)


def f1_score(labels, prediction, end_token, **unused_kwargs):
    """F1 score computation between labels and predictions.

    Args:
        predictions: tensor, model predictions
        labels: tensor, gold output.

    Returns:
        rouge2_fscore: approx rouge-2 f1 score.
    """
    rouge_2_f_score = tf.py_func(compute_f1, (labels, prediction, end_token), tf.float32)
    return tf.metrics.mean(rouge_2_f_score)

