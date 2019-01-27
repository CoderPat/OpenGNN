from opengnn.inputters.token_embedder import SubtokenEmbedder
import tensorflow as tf


def subtokenizer(token):
    return token.split()


def test():
    inputter = SubtokenEmbedder(
        subtokenizer=subtokenizer,
        vocabulary_file_key=None,
        embedding_size=6)
    
    inputter.vocabulary_size = 6
    inputter.vocabulary = tf.contrib.lookup.index_table_from_tensor(["a", "b", "c", "d", "e"],
            num_oov_buckets=1)
    embeddings = tf.get_variable(
        "t_embs", shape=[6, 6], dtype=tf.float32)

    embedding_assign = embeddings.assign(
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]
    )

    sess = tf.Session()
    
    ext, tps, shps = inputter.extract_tensors()

    def generator():
        l = [["a b", "c"], ["d e", "f"]]
        for ele in l:
            yield ext(ele)
    tf.get_variable_scope().reuse_variables()
    dataset = tf.data.Dataset.from_generator(generator, tps, shps)
    dataset = dataset.map(inputter.process)
    dataset = inputter.batch(dataset, 2)
    iter = dataset.make_initializable_iterator()
    ele = iter.get_next()
    t_embs = inputter.transform((ele['features'], ele['length']), mode=tf.estimator.ModeKeys.TRAIN)
    print(sess.run([iter.initializer, embedding_assign, tf.initialize_all_tables()]))
    print(sess.run(t_embs))

if __name__ == "__main__":
    test()