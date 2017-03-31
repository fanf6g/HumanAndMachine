import numpy as np
import pandas as pd
from networkx.algorithms.matching import max_weight_matching
from networkx.convert_matrix import from_numpy_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def minimize(pairs, weights):
    idx = np.argsort(-weights)
    chk_pairs = pairs[idx]
    low = 0
    high = len(chk_pairs) - 1

    while low < high:
        mid = (low + high) / 2
        p = chk_pairs[mid]
        if (p[0] == p[1]):
            low = mid
        else:
            high = mid
        print(p)

        pass
    # chk_pairs[]
    pass


if __name__ == "__main__":
    NAME = 'name'
    ADDR = 'addr'
    CITY = 'city'
    TYPE = 'type'
    CLASS = 'class'
    cv = TfidfVectorizer(ngram_range=(1, 2), dtype='int16', stop_words='english')
    df = pd.read_csv('fz-nophone.csv')

    print(df.head())
    ids = df.pop(CLASS)

    txt = [' '.join(data.map(lambda x: str(x))) for index, data in df.iterrows()]
    cv.fit(txt)
    f_txt = cv.transform(txt)

    m_sim = f_txt.dot(f_txt.T).toarray()

    m = np.triu(m_sim * (m_sim >= 0.7), k=1)

    G = from_numpy_matrix(m)
    # print(G.edges())
    print('G is completed!')

    max_w = max_weight_matching(G)

    pairs = np.array([(ids[i], ids[j]) for (i, j) in max_w.items() if i < j])
    weights = np.array([G[i][j]['weight'] for (i, j) in max_w.items() if i < j])
    print(len(pairs))

    print([(p, w) for p, w in zip(pairs, weights)])

    match = [(a, b) for (a, b) in pairs if a == b]
    print(len(match))

    idx = np.argsort(-weights)
    minimize(pairs, weights)
    # idx = np.argsort(-weiths)
    # print(pairs[idx])
    # print(weiths[idx])

    # match = [pair for pair in pairs if pair in groundTrue]
    # print(len(match))
