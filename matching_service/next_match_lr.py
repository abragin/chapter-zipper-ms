import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import functools


lin_cl = LogisticRegression()
lin_cl.coef_ = np.array([[-1.00901673, -3.36704614, -2.5773804 , -1.3866466 ]])
lin_cl.intercept_ = np.array([-3.36571111])
lin_cl.classes_  = np.array([0,1])
scaler = StandardScaler()
scaler.mean_ = np.array([0.27582206, 0.65347789, 2.85648199, 0.75145329])
scaler.scale_ = np.array([0.22616347, 0.68028582, 1.06506495, 0.78447681])

# Culminative weights
def c_weights(weights):
    res = [weights[0]]
    for w in weights[1:]:
        res.append(res[-1] + w)
    return res

def min_next_distances(s_weight, t_weight, next_weights_source, next_weights_target):
    s_positions = c_weights([s_weight] + next_weights_source)[1:]
    t_positions = c_weights([t_weight] + next_weights_target)[1:]
    t_pointer = 0
    res = []
    for s_pointer in range(len(s_positions)):
        while (s_positions[s_pointer] > t_positions[t_pointer]) and (t_pointer < len(t_positions) - 1):
            t_pointer += 1
        if t_pointer > 0:
            res.append(min(
                abs(s_positions[s_pointer] - t_positions[t_pointer]),
                abs(s_positions[s_pointer] - t_positions[t_pointer-1]),

            ))
        else:
            res.append(abs(s_positions[s_pointer] - t_positions[t_pointer]))

    return res

def min_dist_source_to_target(row):
    return min_next_distances(
        row['s_weight'],
        row['t_weight'],
        row['next_weights_source'],
        row['next_weights_target']
    )

def min_dist_target_to_source(row):
    return min_next_distances(
        row['t_weight'],
        row['s_weight'],
        row['next_weights_target'],
        row['next_weights_source']
    )

def paragraphs_to_weights(ps):
    word_count = sum([len(p.split()) for p in ps])
    return [
        len(p.split())/word_count
        for p in ps
    ]

def processed_distances(row):
    if len(row['min_dist_source_to_target']) == 0:
        print(row)
    dist = max(
        row['min_dist_source_to_target'][0],
        row['min_dist_target_to_source'][0]
    )
    norm_coef = 1/max(
        row['s_weight'] + row['next_weights_source'][0],
        row['t_weight'] + row['next_weights_target'][0],
    )
    return (dist * norm_coef)

def get_lr_prob(ps_source_after, ps_target_after, si, ti):
    s_weights = paragraphs_to_weights(ps_source_after)
    t_weights = paragraphs_to_weights(ps_target_after)

    row = {
        's_weight': sum(s_weights[:si]),
        't_weight': max(sum(t_weights[:ti]), 0.00001),
        'next_weights_source': s_weights[si:si+6],
        'next_weights_target': t_weights[ti:ti+6],
    }

    row['min_dist_source_to_target'] = min_dist_source_to_target(row)
    row['min_dist_target_to_source'] = min_dist_target_to_source(row)
    x = np.array([
        processed_distances(row),
        abs(np.log(row['s_weight']/row['t_weight'])),
        si+ti,
        abs(si-ti)
    ]).reshape(1,4)
    x_scaled = scaler.transform(x)
    return lin_cl.predict_proba(x_scaled)[0,1]


def next_paragraph_match_lr(ps_source_after, ps_target_after):
    max_len = 8
    s_weights = paragraphs_to_weights(ps_source_after)
    t_weights = paragraphs_to_weights(ps_target_after)

    @functools.lru_cache()
    def calc_prob(si, ti):
        row = {
            's_weight': sum(s_weights[:si]),
            't_weight': max(sum(t_weights[:ti]), 0.00001),
            'next_weights_source': s_weights[si:si+6],
            'next_weights_target': t_weights[ti:ti+6],
        }
        row['min_dist_source_to_target'] = min_dist_source_to_target(row)
        row['min_dist_target_to_source'] = min_dist_target_to_source(row)

        x = np.array([
            processed_distances(row),
            abs(np.log(row['s_weight']/row['t_weight'])),
            si+ti,
            abs(si-ti)
        ]).reshape(1,4)
        x_scaled = scaler.transform(x)
        return lin_cl.predict_proba(x_scaled)[0,1]

    si_ti_s = [
        (si, ti)
        for total_len in range(2, max_len+1)
        for si in range(1, total_len)
        if (si < len(ps_source_after)) and ((ti := total_len - si) < len(ps_target_after))
    ]
    for si, ti in si_ti_s:
        match_prob = calc_prob(si,ti)
        merge_s_match_prob = calc_prob(si + 1, ti) if si < len(ps_source_after) - 1 else 0.0
        merge_t_match_prob = calc_prob(si , ti+1) if ti < len(ps_target_after) - 1 else 0.0
        max_next_prob = max(merge_s_match_prob, merge_t_match_prob)
        global match_pred
        # match_pred = lambda mp, mnp: mp > max(mnp, 0.3)
        if match_prob > max(max_next_prob, 0.23):
            return (si, ti)
    if max_len >= len(ps_source_after) + len(ps_target_after):
        return (len(ps_source_after), len(ps_target_after))
