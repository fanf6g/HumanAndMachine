import logging
from itertools import product

import numpy as np
import pandas as pd
import pymongo
import tensorflow as tf
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

NUM_FEATURE = 10


def docs2vecs(docs1, docs2):
    docs = docs1.copy()
    docs.extend(docs2)
    model = Doc2Vec(docs, size=NUM_FEATURE, window=3, min_count=2, workers=4, seed=17)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2v1 = [model.infer_vector(d.words) for d in docs1]
    d2v2 = [model.infer_vector(d.words) for d in docs2]
    ids1 = [str(d.tags) for d in docs1]
    ids2 = [str(d.tags) for d in docs2]
    return pd.DataFrame(np.array(d2v1), index=ids1), pd.DataFrame(np.array(d2v2), index=ids2)


def loadIDs(path, idAbt, idBuy):
    pairs = []
    sa = set(idAbt)
    sb = set(idBuy)
    nnz = 0
    with open(path) as f:
        for line in f.readlines():
            ida, idb = line.split(',')
            ida = ida.strip()
            idb = idb.strip()
            if (ida not in sa and idb not in sb):
                nnz = nnz + 1

            else:
                tmp = '_'.join([ida, idb])
                pairs.append(tmp)

    print(nnz)
    return set(pairs)


def catesian(l1, l2):
    pairs = set([str(i1) + '_' + str(i2) for (i1, i2) in product(l1, l2)])
    return pairs


def pair_partition(set_all, set_pos, set_inc):
    set_neg = set_all.difference(set_pos.union(set_inc))
    allsample_neg = list(set_neg)
    allsample_neg.sort()
    np.random.seed(19)
    sample_neg = list(np.random.choice(allsample_neg, 1000, replace=False))
    sample_neg.sort()
    sample_pos = list(set_pos)
    sample_pos.sort()
    sample_inc = list(set_inc)
    sample_inc.sort()

    return sample_pos, sample_neg, sample_inc


def featuring(samples, vecs_abt, vecs_buy):
    ids_pos = np.array([pair.split('_') for pair in samples])

    ids_valid = np.array([(ida, idb) for (ida, idb) in ids_pos if ida in id_abt and idb in id_buy])

    print(set(ids_valid[:, 0]).difference(id_abt))
    print(set(ids_valid[:, 1]).difference(id_buy))

    pos_abt = vecs_abt.loc[ids_valid[:, 0]].values
    pos_buy = vecs_buy.loc[ids_valid[:, 1]].values

    ids = [ida + '_' + idb for (ida, idb) in ids_valid]
    feature_pos = np.concatenate([pos_abt, pos_buy], axis=1)

    return ids, feature_pos


def groundTrue():
    pass


def evaluate_dnn(train_data, train_label, test_data, test_label):
    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=NUM_FEATURE * 2)]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 20, 10],
                                                n_classes=2,
                                                # model_dir="/tmp/iris_model"
                                                )
    train_y = np.zeros((len(train_label), 2), dtype='int')
    train_y[np.arange(len(train_label)), train_label] = 1
    classifier.fit(x=train_data, y=train_label, steps=2000)

    # Evaluate accuracy.
    test_y = np.zeros((len(test_label), 2), dtype='int')
    test_y[np.arange(len(test_label)), test_label] = 1
    accuracy_score = classifier.evaluate(x=test_data,
                                         y=test_label)["accuracy"]
    print('Accuracy: {0:f}'.format(accuracy_score))


def evaluate_softmax(train_data, train_label, test_data, test_label):
    # Create the model
    NUM_CLASS = 2
    x = tf.placeholder(tf.float32, [None, NUM_FEATURE])
    W = tf.Variable(tf.zeros([NUM_FEATURE, NUM_CLASS]))
    b = tf.Variable(tf.zeros([NUM_CLASS]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASS])

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    L = len(train_data)
    tf.global_variables_initializer().run()
    # Train
    for _ in range(5000):
        idx = np.random.choice(np.arange(L, dtype='int'), 100)
        batch_xs = train_data[idx]
        batch_ys = np.zeros((100, 2), dtype='int')
        batch_ys[np.arange(100), train_label[idx]] = 1
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(sess.run(b))
        # sess.run([y])

    # Test trained model
    test_ys = np.zeros((len(test_data), 2), dtype='int')
    test_ys[np.arange(len(test_data)), test_label] = 1

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_data, y_: test_ys}))


if __name__ == "__main__":
    cv_abt = CountVectorizer(ngram_range=(1, 2), dtype='int16', stop_words='english')
    cv_buy = CountVectorizer(ngram_range=(1, 2), dtype='int16', stop_words='english')
    client = pymongo.MongoClient('localhost', 27017)

    ABTBUY = 'abtbuy'
    ABT = 'abt'
    BUY = 'buy'
    MAPPING = 'mapping'

    ID = 'id'
    NAME = 'name'
    DESCRIPTION = 'description'

    IDABT = 'idAbt'
    IDBUY = 'idBuy'
    IDX = 'index'

    db = client[ABTBUY]

    tbl_abt = db.get_collection(ABT)
    tbl_buy = db.get_collection(BUY)
    tbl_mapping = db.get_collection(MAPPING)

    df_abt = pd.DataFrame(list(tbl_abt.find()))[[ID, NAME, DESCRIPTION]]
    df_abt.set_index([ID])
    print(df_abt.head())
    df_buy = pd.DataFrame(list(tbl_buy.find()))[[ID, NAME, DESCRIPTION]]
    df_buy.set_index(ID)
    df_mapping = pd.DataFrame(list(tbl_mapping.find()))[[IDABT, IDBUY]]

    id_abt = [str(x) for x in df_abt[ID].tolist()]
    id_buy = [str(x) for x in df_buy[ID].tolist()]

    goundTruth = set(
        ['_'.join([str(a), str(b)]) for a, b in zip(df_mapping[IDABT].tolist(), df_mapping[IDBUY].tolist())])
    # print(goundTruth)

    txt_abt = [' '.join(data.map(lambda x: str(x))).lower() for index, data in df_abt[[NAME, DESCRIPTION]].iterrows()]
    txt_buy = [' '.join(data.map(lambda x: str(x))).lower() for index, data in df_buy[[NAME, DESCRIPTION]].iterrows()]

    words_abt = [''.join(c for c in txt.lower() if c.isprintable() and c.isalnum() or c.isspace()).split(' ') for txt
                 in txt_abt]
    words_buy = [''.join(c for c in txt.lower() if c.isprintable() and c.isalnum() or c.isspace()).split(' ') for txt
                 in txt_buy]

    tagDocsAbt = [TaggedDocument(words, str(doc_id)) for doc_id, words in zip(id_abt, words_abt)]
    tagDocsBuy = [TaggedDocument(words, str(doc_id)) for doc_id, words in zip(id_buy, words_buy)]

    vecs_abt, vecs_buy = docs2vecs(tagDocsAbt, tagDocsBuy)
    # vecs_buy = docs2vecs(tagDocsBuy)

    set_pos = loadIDs('xu/abtbuy_pos.csv', id_abt, id_buy)
    set_inc = loadIDs('xu/abtbuy_inc.csv', id_abt, id_buy)

    set_all = catesian(id_abt, id_buy)

    sample_pos, sample_neg, sample_inc = pair_partition(set_all, set_pos, set_inc)

    ids_pos, feature_pos = featuring(sample_pos, vecs_abt, vecs_buy)
    label_pos = np.ones(len(feature_pos), dtype='int')

    ids_neg, feature_neg = featuring(sample_neg, vecs_abt, vecs_buy)
    label_neg = np.zeros(len(feature_neg), dtype='int')

    ids_inc, feautre_inc = featuring(sample_inc, vecs_abt, vecs_buy)

    data = np.concatenate([feature_pos, feature_neg])
    label = np.concatenate([label_pos, label_neg])

    np.random.seed(17)
    randIdx = np.arange(len(feature_pos) + len(feature_neg))
    np.random.shuffle(randIdx)

    print(randIdx)
    print(len(randIdx))

    label_inc = np.array([1 if inc in goundTruth else 0 for inc in sample_inc], dtype='int')
    print(label_inc.sum(), len(sample_inc))

    train_data = data[randIdx]
    train_label = label[randIdx]

    test_data = feautre_inc
    test_label = label_inc

    np.random.seed(17)
    randIdx_test = np.arange(len(feautre_inc))
    np.random.shuffle(randIdx_test)

    evaluate_dnn(train_data, train_label, test_data[randIdx_test], test_label[randIdx_test])

    print(vecs_abt.shape)
    print(vecs_buy.shape)
