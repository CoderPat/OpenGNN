# OpenGNN

OpenGNN is a machine learning library for learning over graph-structured data. It was built with generality in mind and supports tasks such as:

* graph regression
* graph-to-sequence mapping

It supports various graph encoders including GGNNs, GCNs, SequenceGNNs and other variations of [neural graph message passing](https://arxiv.org/pdf/1704.01212.pdf).

This library's design and usage patterns are inspired from [OpenNMT](https://github.com/OpenNMT/OpenNMT-tf) and uses the recent [Dataset](https://www.tensorflow.org/programmers_guide/datasets) and [Estimator](https://www.tensorflow.org/programmers_guide/estimators) APIs.

## Installation

OpenGNN requires 

* Python (>= 3.5)
* Tensorflow (>= 1.10 < 2.0)

To install the library aswell as the command-line entry points run

``` pip install -e .```

## Getting Started

To experiment with the library, you can use one datasets provided in the [data](/data) folder.
For example, to experiment with the chemical dataset, first install the `rdkit` library that 
can be obtained by running `conda install -c rdkit rdkit`.
Then, in the [data/chem](/data/chem) folder, run `python get_data.py` to download the dataset.

After getting the data, generate a node and edge vocabulary for them using 
```bash
ognn-build-vocab --field_name node_labels --save_vocab node.vocab \
                 molecules_graphs_train.jsonl
ognn-build-vocab --no_pad_token --field_name edges --string_index 0 --save_vocab edge.vocab \
                 molecules_graphs_train.jsonl
```

### Command Line

The main entry point to the library is the `ognn-main` command

```bash
ognn-main <run_type> --model_type <model> --config <config_file.yml>
```

Currently there are two run types: `train_and_eval` and `infer`

For example, to train a model on the previously extracted chemical data
(again inside [data/chem](/data/chem)) using a predefined model in the 
catalog

```bash
ognn-main train_and_eval --model_type chemModel --config config.yml
```

You can also define your own model in a custom python script with a `model` function.
For example, we can train using the a custom model in `model.py` using

```bash
ognn-main train_and_eval --model model.py --config config.yml
```

While the training script doesn't log the training to the standard output, 
we can monitor training by using tensorboard on the model directory defined in
[data/chem/config.yml](data/chem/config.yml).

After training, we can perform inference on the valid file running

```
ognn-main infer --model_type chemModel --config config.yml \
                --features_file molecules_graphs_valid.jsonl
                --prediction_file molecules_predicted_valid.jsonl
```


Examples of other config files can be found in the [data](/data) folder.

### Library

The library can also be easily integrated in your own code.
The following example shows how to create a GGNN Encoder to encode a batch of random graphs.

```python
import tensorflow as tf
import opengnn as ognn

tf.enable_eager_execution()

# build a batch of graphs with random initial features
edges = tf.SparseTensor(
    indices=[
        [0, 0, 0, 1], [0, 0, 1, 2],
        [1, 0, 0, 0],
        [2, 0, 1, 0], [2, 0, 2, 1], [2, 0, 3, 2], [2, 0, 4, 3]],
    values=[1, 1, 1, 1, 1, 1, 1],
    dense_shape=[3, 1, 5, 5])
node_features = tf.random_uniform((3, 5, 256))
graph_sizes = [3, 1, 5]

encoder = ognn.encoders.GGNNEncoder(1, 256)
outputs, state = encoder(
    edges,
    node_features,
    graph_sizes)

print(outputs)
```

Graphs are represented by a sparse adjency matrix with dimensionality 
`num_edge_types x num_nodes x num_nodes` and an initial distributed representation for each node.

Similarly to sequences, when batching we need to pad the graphs to the maximum number of nodes in a graph


## Acknowledgments
The design of the library and implementations are based on 
* [OpenNMT-tf](https://github.com/OpenNMT/OpenNMT-tf)
* [Gated Graph Neural Networks](https://github.com/Microsoft/gated-graph-neural-network-samples)

Since most of the code adapted from OpenNMT-tf is spread across multiple files, the license for the
library is located in the [base folder](/OPENNMT.LICENSE) rather than in the headers of the files.

## Reference

If you use this library in your own research, please cite

```
@inproceedings{
    pfernandes2018structsumm,
    title="Structured Neural Summarization",
    author={Patrick Fernandes and Miltiadis Allamanis and Marc Brockschmidt },
    booktitle={Proceedings of the 7th International Conference on Learning Representations (ICLR)},
    year={2019},
    url={https://arxiv.org/abs/1811.01824},
}
```






