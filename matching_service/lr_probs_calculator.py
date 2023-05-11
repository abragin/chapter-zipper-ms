import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# convert an array of paragraph texts to weights array, so that
# each weight is <number of words in paragraph>/<number of words in chapter>
def get_weights(ps):
    ps_ls = [len(p.split()) for p in ps]
    len_total = sum(ps_ls)
    return [max(pl/len_total, 1e-10) for pl in ps_ls]

# Culminative weights
def c_weights(weights):
    res = [weights[0]]
    for w in weights[1:]:
        res.append(res[-1] + w)
    return res

# Original model was trained to answer the question:
# does merging 2 head paragraphs from `source` or`target` improves matching or not.
# So it returned 2 pairs of probabilities:
# [P(2 head source paragraphs should not be merged), P(they should be merged)] - p_m_sa here
# and equivalent pair for target - p_m_ta
# this function converts this probabilities into [p(match), p(source_ahead), p(target_ahead)]
def convert_probs(p_m_sa, p_m_ta):
    if p_m_sa == 0:
        p_m_sa = 1.0e-10
    if p_m_ta == 0:
        p_m_ta = 1.0e-10
    p_m = 1.0 / (1 + (1-p_m_sa)/p_m_sa + (1-p_m_ta)/p_m_ta)
    p_ta = p_m * ((1-p_m_sa)/p_m_sa)
    p_sa = p_m * ((1-p_m_ta)/p_m_ta)
    return [p_m, p_sa, p_ta]

# weights after first 2 paragraphs are merged into one.
def weights_after_merge(weights):
    return [weights[0] + weights[1]] + weights[2:]

# Distances between source_weights and target_weights used
# in linear classifier
def lr_input(s_ws, t_ws):
    s_ws_r = s_ws[1:5]
    t_ws_r = t_ws[1:5]
    if s_ws_r == []:
        s_ws_r = [1.0e-8]
    if t_ws_r == []:
        t_ws_r = [1.0e-8]
    if len(s_ws_r) < len(t_ws_r):
        t_ws_r[len(s_ws_r)-1:] = [sum(t_ws_r[len(s_ws_r)-1:])]
    if len(t_ws_r) < len(s_ws_r):
        s_ws_r[len(t_ws_r)-1:] = [sum(s_ws_r[len(t_ws_r)-1:])]
    w_distance = abs(np.log(
        s_ws[0] / t_ws[0]
    ))
    w_distance_after= np.mean([
        abs(np.log(sw/tw)) for sw, tw in
        zip(s_ws_r, t_ws_r)
    ])
    c_s_ws = c_weights(s_ws[:6])
    c_t_ws = c_weights(t_ws[:6])
    c_distance_after = np.mean([
        abs(sw-tw) for sw, tw in zip(c_s_ws, c_t_ws)
    ])
    return (w_distance, w_distance_after, c_distance_after)

# if based on source/target length only one option is applicable,
# than it should be returned
def handle_edge_cases(ps_source, ps_target):
    if (len(ps_source) < 2) and(len(ps_target) < 2):
        return {
            'match': 1.0,
            'source_ahead': 0.0,
            'target_ahead': 0.0,
            'inconsistent': 0.0
        }
    elif len(ps_source) < 2:
        return {
            'match': 0.0,
            'source_ahead': 1.0,
            'target_ahead': 0.0,
            'inconsistent': 0.0
        }
    elif len(ps_target) < 2:
        return {
            'match': 0.0,
            'source_ahead': 0.0,
            'target_ahead': 1.0,
            'inconsistent': 0.0
        }


class LRProbsCalculator(object):
    def __init__(self, scaler, clf):
        self.scaler = scaler
        self.clf = clf

    def __call__(self, ps_source, ps_target):
        res = handle_edge_cases(ps_source, ps_target)
        if res:
            return res
        else:
            return self.get_probs(ps_source, ps_target)

    def get_probs(self, ps_source, ps_target):
        source_weights = get_weights(ps_source)
        target_weights = get_weights(ps_target)
        (w_distance, w_distance_after, c_distance_after) = lr_input(
            source_weights, target_weights
        )
        input_source_merge = lr_input(
            weights_after_merge(source_weights), target_weights
        )
        input_target_merge = lr_input(
            source_weights, weights_after_merge(target_weights)
        )
        X = pd.DataFrame(data={
            'weight_dist_par': [w_distance, w_distance],
            'weight_dist_next': [w_distance_after, w_distance_after],
            'c_weight_dist_next': [c_distance_after, c_distance_after],
            'weight_dist_par_after_s_merge': [input_source_merge[0], input_target_merge[0]],
            'weight_dist_next_after_s_merge': [input_source_merge[1], input_target_merge[1]],
            'c_weight_dist_next_after_s_merge': [input_source_merge[2], input_target_merge[2]],
        })
        x_scaled = self.scaler.transform(X)
        pred = self.clf.predict_proba(x_scaled)
        probs = convert_probs(pred[0][0], pred[1][0])
        return {
            'match': probs[0],
            'source_ahead': probs[1],
            'target_ahead': probs[2],
            'inconsistent': 0.0
        }

scaler =  StandardScaler()
scaler.mean_ = np.array([
      0.1860575 , 0.28707439, 0.01254617, 0.82783402, 1.26658851, 0.02764434
    ])
scaler.scale_ = np.array([
      0.24838857, 0.33084664, 0.01216887, 0.65755263, 2.17709204, 0.02135905
    ])
scaler.feature_names_in_ = [
    'weight_dist_par', 'weight_dist_next', 'c_weight_dist_next',
    'weight_dist_par_after_s_merge', 'weight_dist_next_after_s_merge',
    'c_weight_dist_next_after_s_merge'
]
lin_cl = LogisticRegression()
lin_cl.coef_ = np.array([[
      1.57098274,  0.58236663,  0.89874209,
      -3.76852604, -2.00030208, -1.06507971
    ]])
lin_cl.intercept_ = np.array([-8.68506419])
lin_cl.classes_  = np.array([0,1])

get_lr_probs = LRProbsCalculator(scaler, clf=lin_cl)
