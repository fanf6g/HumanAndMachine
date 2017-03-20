import logging
from itertools import product

import numpy as np
import pandas as pd
import pymongo
import tensorflow as tf
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def docs2vecs(docs):
    model = Doc2Vec(docs, size=100, window=5, min_count=3, workers=4)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2v = [model.infer_vector(d.words) for d in docs]
    ids = [str(d.tags) for d in docs]
    return pd.DataFrame(np.array(d2v), index=ids)


def loadIDs(path):
    pairs = []
    with open(path) as f:
        for line in f.readlines():
            ids = line.split(',')
            tmp = '_'.join(map(lambda x: str(x).strip(), ids))
            pairs.append(tmp)

    return set(pairs)


def catesian(l1, l2):
    pairs = set([str(i1) + '_' + str(i2) for (i1, i2) in product(l1, l2)])
    return pairs


def pair_partition(set_all, set_pos, set_inc):
    set_neg = set_all.difference(set_pos.union(set_inc))
    allsample_neg = list(set_neg)
    allsample_neg.sort()
    np.random.seed(17)
    sample_neg = list(np.random.choice(allsample_neg, 1000, replace=False))
    sample_neg.sort()
    sample_pos = list(set_pos)
    sample_pos.sort()
    sample_inc = list(set_inc)
    sample_inc.sort()

    return sample_pos, sample_neg, sample_inc


def groundTrue():
    pass


def evaluate():
    with tf.Session() as sess:
        pass


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

    id_abt = df_abt[ID].tolist()
    id_buy = df_buy[ID].tolist()

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

    vecs_abt = docs2vecs(tagDocsAbt)
    vecs_buy = docs2vecs(tagDocsBuy)

    set_pos = loadIDs('abtbuy_consist.txt')
    set_inc = loadIDs('abtbuy_inconsist.txt')

    set_all = catesian(id_abt, id_buy)

    sample_pos, sample_neg, sample_inc = pair_partition(set_all, set_pos, set_inc)

    # print(sample_pos)
    # print(sample_neg)
    # print(sample_inc)

    # print(vecs_abt.shape)
    # print(vecs_buy.shape)
