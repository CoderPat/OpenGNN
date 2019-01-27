from opengnn.models.graph_to_sequence import GraphToSequence
from opengnn.models.graph_regressor import GraphRegressor

from opengnn.encoders.ggnn_encoder import GGNNEncoder
from opengnn.encoders.gcn_encoder import GCNEncoder
from opengnn.decoders.sequence.rnn_decoder import RNNDecoder
from opengnn.decoders.sequence.hybrid_pointer_decoder import HybridPointerDecoder
from opengnn.inputters.token_embedder import TokenEmbedder, SubtokenEmbedder
from opengnn.inputters.copying_token_embedder import CopyingTokenEmbedder
from opengnn.inputters.features_inputter import FeaturesInputter
from opengnn.inputters.graph_inputter import GraphInputter, GraphEmbedder


class chemModelGGNN(GraphRegressor):
    def __init__(self):
        super().__init__(
            source_inputter=GraphEmbedder(
                edge_vocabulary_file_key="edge_vocabulary",
                node_embedder=TokenEmbedder(
                    vocabulary_file_key="node_vocabulary",
                    embedding_size=64)),
            target_inputter=FeaturesInputter(),
            encoder=GGNNEncoder(
                num_timesteps=[2, 2],
                node_feature_size=64),
            name="chemModelGGNN")


class chemModelGCN(GraphRegressor):
    def __init__(self):
        super().__init__(
            source_inputter=GraphEmbedder(
                edge_vocabulary_file_key="edge_vocabulary",
                node_embedder=TokenEmbedder(
                    vocabulary_file_key="node_vocabulary",
                    embedding_size=64)),
            target_inputter=FeaturesInputter(),
            encoder=GCNEncoder(
                layer_sizes=[64, 32, 16]),
            name="chemModelGCN")
