import numpy as np
import pandas as pd
import pymongo
from networkx.algorithms.matching import max_weight_matching
from networkx.convert_matrix import from_numpy_matrix
from scipy.linalg import block_diag
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == "__main__":
    cv = TfidfVectorizer(ngram_range=(1, 1), dtype='int16', stop_words='english')
    # cv_buy = CountVectorizer(ngram_range=(1, 2), dtype='int16', stop_words='english')
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
    id_abt = df_abt.pop(ID).values.tolist()

    df_buy = pd.DataFrame(list(tbl_buy.find()))[[ID, NAME, DESCRIPTION]]
    id_buy = df_buy.pop(ID).values.tolist()

    ids = []
    ids.extend(id_abt)
    ids.extend(id_buy)

    df_mapping = pd.DataFrame(list(tbl_mapping.find()))[[IDABT, IDBUY]]
    groundTrue = [str(a) + '_' + str(b) for (a, b) in df_mapping.values]

    print(df_abt.columns)
    print(df_buy.columns)
    print(df_mapping.columns)

    txt_abt = [' '.join(data.map(lambda x: str(x))) for index, data in df_abt[[NAME, DESCRIPTION]].iterrows()]
    txt_buy = [' '.join(data.map(lambda x: str(x))) for index, data in df_buy[[NAME, DESCRIPTION]].iterrows()]

    txt = []
    txt.extend(txt_abt)
    txt.extend(txt_buy)

    cv.fit(txt)

    f_txt = cv.transform(txt)

    m_sim = f_txt.dot(f_txt.T).toarray()

    # N, labels = connected_components(m_sim >= 0.3, directed=False, return_labels=True)

    abt_ones = np.ones((len(id_abt), len(id_abt)))
    buy_ones = np.ones((len(id_buy), len(id_buy)))

    mask = -block_diag(abt_ones, buy_ones) + 1
    print(mask)

    # 计算上三角矩阵
    m = np.triu(m_sim * (m_sim >= 0.2), k=1) * mask
    print(m)
    G = from_numpy_matrix(m)
    # print(G.edges())
    print('G is completed!')

    max_w = max_weight_matching(G)

    # pair1 = [(str(i) + '_' + str(j)) for (i, j) in max_w.items() if i < j]
    pairs = np.array([(str(ids[i]) + '_' + str(ids[j])) for (i, j) in max_w.items() if i < j])
    weiths = np.array([G[i][j]['weight'] for (i, j) in max_w.items() if i < j])
    print(len(pairs))

    print([(p, w) for p, w in zip(pairs, weiths)])
    # idx = np.argsort(-weiths)
    # print(pairs[idx])
    # print(weiths[idx])

    match = [pair for pair in pairs if pair in groundTrue]
    print(len(match))
    # print(pairs)
    # print(weiths)
    # maximal_matching(G)
    # print(len(G[1318]))

    # print(np.unique(labels, return_counts=True))
    #
    # sc = SpectralClustering(n_clusters=500, n_neighbors=3)
    # y = sc.fit_predict(f_txt)
    #
    # print(np.unique(y, return_counts=True))
