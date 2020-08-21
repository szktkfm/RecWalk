import networkx as nx
from networkx.exception import NetworkXError

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle

from SLIM_model import SLIM
import optuna
import time

import warnings
warnings.filterwarnings('ignore')


# dataload
slim_train = pd.read_csv('../data_luxury_5core/bpr/user_item_train.csv')
triplet_df = pd.read_csv('../data_luxury_5core/triplet.csv')
edges = [[r[0], r[1]] for r in triplet_df.values]
# user-itemとitem-userどちらの辺も追加
for r in triplet_df.values:
    if r[2] == 0:
        edges.append([r[1], r[0]])

user_list = []
item_list = []
entity_list = []
with open('../data_luxury_5core/user_list.txt', 'r') as f:
    for l in f:
        user_list.append(l.replace('\n', ''))
with open('../data_luxury_5core/item_list.txt', 'r') as f:
    for l in f:
        item_list.append(l.replace('\n', ''))
with open('../data_luxury_5core/entity_list.txt', 'r') as f:
    for l in f:
        entity_list.append(l.replace('\n', ''))


user_items_test_dict = pickle.load(open('../data_luxury_5core/user_items_test_dict.pickle', 'rb'))

def load_params():
    return pickle.load(open('result/best_param4.pickle', 'rb'))

# SLIMのハイパラをロードする
slim_param = pickle.load(open('best_param_slim.pickle', 'rb'))

# ハイパラ
# gamma
def train_SLIM(hyparam):
    # ハイパラロードもっと上手く書く
    alpha = hyparam['alpha']
    l1_ratio = hyparam['l1_ratio']
    #lin_model = hyparam['lin_model']
    slim = SLIM(alpha, l1_ratio, len(user_list), len(item_list), lin_model='elastic')
    slim.fit_multi(slim_train)
    #slim.load_sim_mat('./sim_mat.txt', slim_train)
    #slim.save_sim_mat('./sim_mat.txt')
    return slim

# load network
G = nx.DiGraph()
G.add_nodes_from([i for i in range(len(entity_list))])
G.add_edges_from(edges)


def mk_sparse_sim_mat(G, item_mat):
    item_mat = scipy.sparse.csr_matrix(item_mat)
    item_len = item_mat.shape[0]
    I = scipy.sparse.eye(len(G.nodes()) - item_len)
    
    M = scipy.sparse.block_diag((item_mat, I))
    #print(M)
    # RecWalk論文の定義
    M_ = np.array(1 - M.sum(axis=1) / np.max(M.sum(axis=1)))
                                    
    M = M / np.max(M.sum(axis=1)) + scipy.sparse.diags(M_.transpose()[0])
    #print(type(M))
    #print(M.shape)
    return M



def pagerank_scipy(G, sim_mat, alpha, beta, personalization=None,
                   max_iter=100, tol=1.0e-6, weight='weight',
                   dangling=None):
    
    #import scipy.sparse

    N = len(G)
    if N == 0:
        return {}

    nodelist = G.nodes()
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        missing = set(nodelist) - set(personalization)
        if missing:
            raise NetworkXError('Personalization vector dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        p = scipy.array([personalization[n] for n in nodelist],
                        dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        missing = set(nodelist) - set(dangling)
        if missing:
            raise NetworkXError('Dangling node dictionary '
                                'must have a value for every node. '
                                'Missing nodes %s' % missing)
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling[n] for n in nodelist],
                                       dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    
    # 遷移行列とsim_matを統合
    #sim_mat = mk_sparse_sim_mat(G, item_mat)
    M = beta * M + (1 - beta) * sim_mat
    #S = scipy.array(M.sum(axis=1)).flatten()
    #S[S != 0] = 1.0 / S[S != 0]
    #Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    #M = Q * M


    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + \
            (1 - alpha) * p
        # check convergence, l1 norm
        x = x / x.sum()
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    # pagerankの収束ちゃんとやっとく
    print(x.sum())
    print(err)
    print(N * tol)
    
    #raise NetworkXError('pagerank_scipy: power iteration failed to converge '
                        #'in %d iterations.' % max_iter)
        
    return dict(zip(nodelist, map(float, x)))




def item_ppr(sim_mat, user, alpha, beta):
    val = np.zeros(len(G.nodes()))
    val[user] = 1
    k = [i for i in range(len(G.nodes()))]
    personal_vec = dict(zip(k, val))
    #print(personal_vec)
    #ppr = pagerank_numpy(G, slim.sim_mat, alpha, beta, personalization=personal_vec)
    ppr = pagerank_scipy(G, sim_mat, alpha, beta, personalization=personal_vec)
    #return pr
    
    # random 後で消す
    # val = np.random.dirichlet([1 for i in range(len(G.nodes))], 1)[0]
    #val = np.random.rand(len(G.nodes()))
    #val /= val.sum()
    #k = [i for i in range(len(G.nodes))]
    #ppr = dict(zip(k, val))
    
    pred = []
    item_idx = [entity_list.index(i) for i in item_list]
    for i in item_idx:
        pred.append(ppr[i])
    
    return pred


def get_ranking_mat(slim, alpha, beta):
    user_idx = [entity_list.index(u) for u in user_list]
    ranking_mat = []
    count = 0
    sim_mat = mk_sparse_sim_mat(G, slim.sim_mat)
    count = 0
    for u in user_idx:
        pred = item_ppr(sim_mat, u, alpha, beta)
        #print(pred[0:5])
        sorted_idx = np.argsort(np.array(pred))[::-1]
        #print(sorted_idx[:10])
        ranking_mat.append(sorted_idx)

        #count += 1
        #if count > 2:
        #   break
            
    return ranking_mat


def topn_precision(ranking_mat, user_items_dict, n=10):
    user_idx = [entity_list.index(u) for u in user_list]
    not_count = 0
    precision_sum = 0
        
    for i in range(len(ranking_mat)):
        if len(user_items_dict[user_idx[i]]) == 0:
            not_count += 1
            continue
        sorted_idx = ranking_mat[i]
        topn_idx = sorted_idx[:n]  
        hit = len(set(topn_idx) & set(user_items_dict[user_idx[i]]))
        precision = hit / len(user_items_dict[user_idx[i]])
        precision_sum += precision
        
    return precision_sum / (len(user_idx) - not_count)

# laod model
slim = train_SLIM(slim_param)

def objective(trial):
    start = time.time()
    # ハイパラ読み込み
    # gamma = trial.suggest_loguniform('gamma', 1e-6, 1e-3)
    # lin_model = trial.suggest_categorical('lin_model', ['lasso', 'elastic'])
    alpha = trial.suggest_uniform('alpha', 0, 1)
    beta = trial.suggest_uniform('beta', 0, 0.5)

    ranking_mat = get_ranking_mat(slim, alpha, beta)
    score = topn_precision(ranking_mat, user_items_test_dict)
    
    mi, sec = time_since(time.time() - start)
    print('{}m{}s'.format(mi, sec))
    
    return -1 * score

def time_since(runtime):
    mi = int(runtime / 60)
    sec = int(runtime - mi * 60)
    return (mi, sec)

if __name__ == '__main__':
    params = load_params()
    alpha = params['alpha']
    beta = params['beta']
    ranking_mat = get_ranking_mat(slim, alpha, beta)
    score = topn_precision(ranking_mat, user_items_test_dict)
    np.savetxt('score.txt', np.array([score]))




    


