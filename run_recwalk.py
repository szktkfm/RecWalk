import networkx as nx
from networkx.exception import NetworkXError

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import pickle

from SLIM_model import SLIM
from dataloader import AmazonDataset
import optuna
import time

from evaluate import Evaluater

import warnings
warnings.filterwarnings('ignore')





# ハイパラ
# gamma
def train_SLIM(hyparam):
    slim_train = pd.read_csv('../data_luxury_5core/valid1/bpr/user_item_train.csv')
    user_list = []
    item_list = []
    with open('./data2/user_list.txt', 'r') as f:
        for l in f:
            user_list.append(l.replace('\n', ''))
    with open('./data2/item_list.txt', 'r') as f:
        for l in f:
            item_list.append(l.replace('\n', ''))

    # ハイパラロードもっと上手く書く
    alpha = hyparam['alpha']
    l1_ratio = hyparam['l1_ratio']
    #lin_model = hyparam['lin_model']
    slim = SLIM(alpha, l1_ratio, len(user_list), len(item_list), lin_model='elastic')
    #slim.fit_multi(slim_train)
    slim.load_sim_mat('./sim_mat.txt', slim_train)
    #slim.save_sim_mat('./sim_mat.txt')
    return slim



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




def item_ppr(G, sim_mat, user, alpha, beta, dataset):
    val = np.zeros(len(G.nodes()))
    val[user] = 1
    k = [i for i in range(len(G.nodes()))]
    personal_vec = dict(zip(k, val))
    #print(personal_vec)
    #ppr = pagerank_scipy(G, sim_mat, alpha, beta, personalization=personal_vec)
    #return pr
    
    # random 後で消す
    # val = np.random.dirichlet([1 for i in range(len(G.nodes))], 1)[0]
    val = np.random.rand(len(G.nodes()))
    val /= val.sum()
    k = [i for i in range(len(G.nodes))]
    ppr = dict(zip(k, val))
    
    pred = []
    for i in dataset.item_idx:
        pred.append(ppr[i])
    
    return pred


def get_ranking_mat(G, slim, alpha, beta, dataset):
    #user_idx = [entity_list.index(u) for u in user_list]
    ranking_mat = []
    count = 0
    sim_mat = mk_sparse_sim_mat(G, slim.sim_mat)
    count = 0
    for u in dataset.user_idx:
        pred = item_ppr(G, sim_mat, u, alpha, beta, dataset)
        #print(pred[0:5])
        sorted_idx = np.argsort(np.array(pred))[::-1]
        #print(sorted_idx[:10])
        ranking_mat.append(sorted_idx)

    return ranking_mat



# SLIMのハイパラをロードする
slim_param = pickle.load(open('best_param_slim.pickle', 'rb'))
# laod model
slim = train_SLIM(slim_param)

def objective(trial):
    start = time.time()
    # ハイパラ読み込み
    # gamma = trial.suggest_loguniform('gamma', 1e-6, 1e-3)
    # lin_model = trial.suggest_categorical('lin_model', ['lasso', 'elastic'])
    alpha = trial.suggest_uniform('alpha', 0, 1)
    beta = trial.suggest_uniform('beta', 0, 0.5)

    data_dirs = ['../data_luxury_5core/valid1/', '../data_luxury_5core/valid2/']
    score_sum = 0
    for data_dir in data_dirs:
        # dataload
        dataset = AmazonDataset(data_dir)

        edges = [[r[0], r[1]] for r in dataset.triplet_df.values]
        # user-itemとitem-userどちらの辺も追加
        for r in dataset.triplet_df.values:
            if r[2] == 0:
                edges.append([r[1], r[0]])
            
        # load network
        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(len(dataset.entity_list))])
        G.add_edges_from(edges)

        evaluater = Evaluater(data_dir)
        ranking_mat = get_ranking_mat(G, slim, alpha, beta, dataset)
        #score = evaluater.topn_precision(ranking_mat)
        score = evaluater.topn_map(ranking_mat)

        score_sum += score
    
    mi, sec = time_since(time.time() - start)
    print('{}m{}s'.format(mi, sec))
    
    return -1 * score_sum / 2

def time_since(runtime):
    mi = int(runtime / 60)
    sec = int(runtime - mi * 60)
    return (mi, sec)

if __name__ == '__main__':

    study = optuna.create_study()
    study.optimize(objective, n_trials=20)
    df = study.trials_dataframe() # pandasのDataFrame形式
    df.to_csv('./hyparams_result_no_item-item_relation.csv')
    # save best params 
    with open('best_param_no_item-item_relation.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)