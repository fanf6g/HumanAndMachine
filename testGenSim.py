import logging

import numpy as np
import pandas as pd
import pymongo
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def docs2vecs(docs):
    taggedDocs = [TaggedDocument(words=words, tags=str(id)) for id, words in enumerate(docs)]
    model = Doc2Vec(taggedDocs, size=100, window=5, min_count=3, workers=4)
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    d2v = [model.infer_vector(d) for d in docs]
    return np.array(d2v)


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
    df_abt.columns = [[IDABT, NAME, DESCRIPTION]]
    df_abt[IDX] = np.arange(df_abt.shape[0])
    df_abt[IDABT + '2'] = df_abt[IDABT]

    df_buy = pd.DataFrame(list(tbl_buy.find()))[[ID, NAME, DESCRIPTION]]
    df_buy.columns = [[IDBUY, NAME, DESCRIPTION]]
    df_buy[IDX] = np.arange(df_buy.shape[0])
    df_buy[IDBUY + '2'] = df_buy[IDBUY]

    df_mapping = pd.DataFrame(list(tbl_mapping.find()))[[IDABT, IDBUY]]

    txt_abt = [' '.join(data.map(lambda x: str(x))).lower() for index, data in df_abt[[NAME, DESCRIPTION]].iterrows()]
    txt_buy = [' '.join(data.map(lambda x: str(x))).lower() for index, data in df_buy[[NAME, DESCRIPTION]].iterrows()]

    txt1_abt = [''.join(c for c in txt.lower() if c.isalnum() or c.isspace()).split(' ') for txt in txt_abt]
    txt1_buy = [''.join(c for c in txt.lower() if c.isalnum() or c.isspace()).split(' ') for txt in txt_buy]

    vecs_abt = docs2vecs(txt1_abt)
    vecs_buy = docs2vecs(txt1_buy)

    print(vecs_abt.shape)
    print(vecs_buy.shape)
