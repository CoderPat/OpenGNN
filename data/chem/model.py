import opengnn as ognn


def model():
    return ognn.models.GraphRegressor(
        source_inputter=ognn.inputters.GraphEmbedder(
            edge_vocabulary_file_key="edge_vocabulary",
            node_embedder=ognn.inputters.TokenEmbedder(
                vocabulary_file_key="node_vocabulary",
                embedding_size=64)),
        target_inputter=ognn.inputters.FeaturesInputter(),
        encoder=ognn.encoders.GGNNEncoder(
            num_timesteps=[2, 2],
            node_feature_size=64),
        name="chemModelCustom")
