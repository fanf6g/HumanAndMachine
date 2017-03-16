import numpy as np
import pandas as pd
import pymongo
from scipy.sparse.csc import csc_matrix
from sklearn.decomposition.pca import PCA
from sklearn.feature_extraction.text import CountVectorizer


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

    print(df_abt.columns)
    print(df_buy.columns)
    print(df_mapping.columns)

    map_buy = df_mapping.set_index(IDBUY).join(df_buy.set_index(IDBUY), how='inner')
    print(map_buy.columns)

    abt_buy_mapp = df_abt.set_index(IDABT).join(map_buy.set_index(IDABT), how='inner', lsuffix='_abt', rsuffix='_buy')

    txt_abt = [' '.join(data.map(lambda x: str(x))) for index, data in df_abt[[NAME, DESCRIPTION]].iterrows()]
    txt_buy = [' '.join(data.map(lambda x: str(x))) for index, data in df_buy[[NAME, DESCRIPTION]].iterrows()]

    n_components = 500

    f_abt = cv_abt.fit_transform(txt_abt)
    pca_abt = PCA(n_components=n_components)
    f_red_abt = pca_abt.fit_transform(f_abt.toarray())
    # print(pca_abt.explained_variance_ratio_)

    f_buy = cv_abt.fit_transform(txt_buy)
    pca_buy = PCA(n_components=n_components)
    f_red_buy = pca_buy.fit_transform(f_buy.toarray())
    # print(pca_buy.explained_variance_ratio_)

    dot_prod = f_red_abt.dot(f_red_buy.T)
    # print(dot_prod.shape)

    row_idx = abt_buy_mapp[IDX + '_abt'].tolist()
    col_idx = abt_buy_mapp[IDX + '_buy'].tolist()

    print(row_idx)
    print(col_idx)

    # match = csc_matrix((np.array([1.] * len(row_idx)), row_idx, col_idx), shape=dot_prod.shape, dtype='int16')
    match = csc_matrix((np.array([1.] * len(row_idx)), (row_idx, col_idx)), shape=dot_prod.shape, dtype='int16')
    print(match.shape, match.nnz)

    # print(df_abt.head())
    # print(df_buy.head())
    # print(df_mapping.head())
    # print(abt_buy_mapp.columns)
    # print(abt_buy_mapp.head())
    # print(abt_buy_mapp.columns)
    # print(abt_buy_mapp.head())
